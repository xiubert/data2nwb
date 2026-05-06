"""
NWB writer for QImaging epifluorescence (.qcamraw) recordings.

Mirrors lib.nwbScanImage.genNWBfromScanImage_pc but uses OnePhotonSeries.
Reuses the existing pulse-metadata fallback ladder unchanged.
"""
import os
import glob
from uuid import uuid4

import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject
from pynwb.ophys import (
    Fluorescence,
    ImageSegmentation,
    OnePhotonSeries,
    OpticalChannel,
    RoiResponseSeries,
)
from hdmf.common import DynamicTable, VectorData
from hdmf.backends.hdf5.h5_utils import H5DataIO

import lib.mat2py
from lib.qcamraw import (
    read_qcamraw,
    header_summary,
    get_qcamraw_start_time,
    load_or_select_roi,
    mean_fluo_in_roi_vectorised,
)
from lib.nwbScanImage import writeNWB  # reuse


def _resolve_qcam_filelist(experimentDir, experimentID):
    """Resolve .qcamraw file list and per-file treatment / type.

    Priority:
      1. {experimentID}_qcamFileList.csv  (columns: file, treatment, type)
      2. glob of {experimentID}*.qcamraw, treatment='none', type='stim'
    """
    listCSV = os.path.join(experimentDir, f'{experimentID}_qcamFileList.csv')
    if os.path.exists(listCSV):
        df = pd.read_csv(listCSV)
        files = df['file'].tolist()
        treatments = (df['treatment'].tolist()
                      if 'treatment' in df.columns
                      else ['none'] * len(files))
        types = (df['type'].tolist()
                 if 'type' in df.columns
                 else ['stim'] * len(files))
        return files, treatments, types

    files = sorted(
        os.path.basename(p) for p in
        glob.glob(os.path.join(experimentDir, f'{experimentID}*.qcamraw'))
    )
    if not files:
        raise FileNotFoundError(
            f'No .qcamraw files found in {experimentDir}')
    return files, ['none'] * len(files), ['stim'] * len(files)


def genNWBfromQcamraw_pc(
        experimentID: str,
        dataPath: str,
        NWBoutputPath: str,
        subject: Subject,
        session_description: str,
        experiment_description: str,
        keywords: list,
        experimenter: str,
        related_publications: str,
        lab: str,
        institution: str,
        # imaging metadata
        imaging_scopeDesc: str,
        imaging_manufacturer: str,
        imaging_opticalChannel0desc: str,
        imaging_emissionLambda: float,
        imagingPlane_rate: float,
        imagingPlane_desc: str,
        imagingPlane_excitationLambda: float,
        imagingPlane_loc: str,
        imagingPlane_indicator: str,
        imagingPlane_gridSpacing: list,
        imagingPlane_gridSpacingUnit: str,
        imagingPlane_originCoords: list,
        imagingPlane_originCoordsUnit: str,
        # qcamraw-specific processing
        qcam_frameRate: float,
        qcam_baseline: list = (2.0, 3.0),
        qcam_filtOrder: int = 4,
        qcam_filtCutoffFreq: float = 5.0,
        qcam_stimlen: float = 0.4,
        qcam_temporalAvgFrameWindow: int = 10,
        qcam_timestamp_field: str = None,   # override header timestamp key
        overWriteNWB: bool = True,
        returnNWB: bool = False,
        ) -> NWBFile:
    """Build an NWB file for one epifluorescence experiment.

    Movies are read once and discarded after their OnePhotonSeries and
    fluorescence trace are written, so peak memory ≈ size of the largest
    single .qcamraw rather than the whole experiment.
    """
    experimentDir = os.path.join(dataPath, experimentID)
    print(f'Generating NWB file for {experimentID} (epifluorescence)')

    if abs(float(qcam_frameRate) - float(imagingPlane_rate)) > 1e-6:
        raise ValueError(
            f'qcam_frameRate ({qcam_frameRate}) must match '
            f'imagingPlane_rate ({imagingPlane_rate}).'
        )

    qcam_files, treatments, file_types = _resolve_qcam_filelist(
        experimentDir, experimentID)

    # Conditions → ImagingPlane / PlaneSegmentation grouping.
    # Single 'none' treatment uses 'all' to match 2P writer convention.
    if set(treatments) == {'none'}:
        conditions = ['all']
        file_conds = ['all'] * len(qcam_files)
    else:
        conditions = sorted(set(treatments))
        file_conds = list(treatments)

    # Session start = earliest acquisition time across recordings.
    starts_abs = [get_qcamraw_start_time(
                    os.path.join(experimentDir, f),
                    header_field=qcam_timestamp_field)
                  for f in qcam_files]
    session_start = min(starts_abs)

    # ---------------------------------------------------- NWBFile shell ----
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=session_start,
        experimenter=[experimenter],
        lab=lab,
        institution=institution,
        experiment_description=experiment_description,
        keywords=keywords,
        related_publications=related_publications,
    )
    nwbfile.subject = subject

    # --------------------------------------------------- device + plane ----
    device = nwbfile.create_device(
        name='Camera',
        description=imaging_scopeDesc,
        manufacturer=imaging_manufacturer,
    )
    optical_channel = OpticalChannel(
        name='OpticalChannel',
        description=imaging_opticalChannel0desc,
        emission_lambda=float(imaging_emissionLambda),
    )

    imgPlane = {}
    for cond in conditions:
        imgPlane[cond] = nwbfile.create_imaging_plane(
            name=f'ImagingPlane_{cond}',
            optical_channel=optical_channel,
            imaging_rate=float(imagingPlane_rate),
            description=imagingPlane_desc,
            device=device,
            excitation_lambda=float(imagingPlane_excitationLambda),
            indicator=imagingPlane_indicator,
            location=imagingPlane_loc,
            grid_spacing=imagingPlane_gridSpacing,
            grid_spacing_unit=imagingPlane_gridSpacingUnit,
            origin_coords=imagingPlane_originCoords,
            origin_coords_unit=imagingPlane_originCoordsUnit,
        )

    # ------------------------------------------ PASS 1: ROI selection ----
    # Read first file of each condition once to draw / load the ROI.
    qcam_cfg = dict(
        baseline=tuple(qcam_baseline),
        filtOrder=qcam_filtOrder,
        filtCutoffFreq=qcam_filtCutoffFreq,
        stimlen=qcam_stimlen,
        temporalAvgFrameWindow=qcam_temporalAvgFrameWindow,
    )
    cond_to_first_idx = {c: file_conds.index(c) for c in conditions}
    masks_per_cond = {}
    for cond, idx in cond_to_first_idx.items():
        movie, _ = read_qcamraw(os.path.join(experimentDir, qcam_files[idx]))
        _, mask, _ = load_or_select_roi(
            os.path.join(experimentDir, qcam_files[idx]),
            movie, qcam_frameRate, qcam_cfg,
            condition=cond)
        masks_per_cond[cond] = mask
        del movie
    print('resolved ROIs')

    # ----------------- ophys module + segmentation + fluorescence ----
    # Both ImageSegmentation and Fluorescence are attached to the file
    # before any RoiResponseSeries / DynamicTableRegion is constructed,
    # so the ancestor-chain validation in HDMF passes cleanly.
    ophys_module = nwbfile.create_processing_module(
        name='ophys',
        description='widefield epifluorescence processed data')

    img_seg = ImageSegmentation()
    ophys_module.add(img_seg)

    fl = Fluorescence()
    ophys_module.add(fl)

    plane_seg = {}
    rt_regions = {}

    # ---- PASS 2: stream each .qcamraw → OnePhotonSeries + trace ----
    starts_rel, nframes = [], []

    for i, (fname, cond) in enumerate(zip(qcam_files, file_conds)):
        path = os.path.join(experimentDir, fname)
        movie, header = read_qcamraw(path)
        hdr_summary = header_summary(header)

        start_rel = (starts_abs[i] - session_start).total_seconds()
        starts_rel.append(start_rel)
        nframes.append(movie.shape[0])

        op = OnePhotonSeries(
            name=f'OnePhotonSeries_{i:03}',
            description='Raw widefield epifluorescence movie',
            data=H5DataIO(data=movie, compression=True),
            imaging_plane=imgPlane[cond],
            rate=float(qcam_frameRate),
            starting_time=float(start_rel),
            unit='n.a.',
            comments=(f'file: {fname}, treatment: {treatments[i]}, '
                      f'type: {file_types[i]}, nFrames: {movie.shape[0]}; '
                      f'{hdr_summary}'),
        )
        nwbfile.add_acquisition(op)

        # compute trace immediately so we can drop the movie
        trace = mean_fluo_in_roi_vectorised(movie,
                                            masks_per_cond[cond])  # (F, 1)
        del movie

        # lazily build per-condition PlaneSegmentation + ROI table region
        # using the first OnePhotonSeries of the condition as reference.
        if cond not in plane_seg:
            plane_seg[cond] = img_seg.create_plane_segmentation(
                name=f'PlaneSegmentation_{cond}',
                description=f'rectangular ROI for {cond}',
                imaging_plane=imgPlane[cond],
                reference_images=[op],
            )
            plane_seg[cond].add_roi(
                image_mask=masks_per_cond[cond].astype(np.uint8))
            rt_regions[cond] = plane_seg[cond].create_roi_table_region(
                region=[0], description=f'ROI for {cond}')

        fl.create_roi_response_series(
            name=f'RoiResponseSeries_{i:03}',
            description=(f'mean ROI fluorescence for '
                         f'OnePhotonSeries_{i:03}'),
            data=trace,                     # (nFrames, 1) — matches 2P writer
            rois=rt_regions[cond],
            unit='n.a.',
            rate=float(qcam_frameRate),
            starting_time=float(start_rel),
        )

    print('added 1P data, ROI segmentation, and fluorescence traces')

    # ----------------------------------- stim/pulse metadata table ----
    _add_stim_table(nwbfile, experimentDir,
                    qcam_files, file_types, treatments,
                    starts_rel, nframes, qcam_frameRate)

    writeNWB(NWBoutputPath, nwbfile, overWrite=overWriteNWB)
    print(f'NWB write success to: {NWBoutputPath}')
    if returnNWB:
        return nwbfile


def _add_stim_table(nwbfile, experimentDir,
                    qcam_files, file_types, treatments,
                    starts_rel, nframes, frame_rate):
    """Add stimulus metadata from pulseLegendQcam.mat / pulseLegendQcam.csv.

    Strict-match behaviour: every pulseFile must correspond to a known
    acquisition file, otherwise raises.
    """
    legendMat = os.path.join(experimentDir, 'pulseLegendQcam.mat')
    legendCSV = os.path.join(experimentDir, 'pulseLegendQcam.csv')

    if os.path.exists(legendMat):
        print('reading pulse metadata from pulseLegendQcam.mat')
        (pulseFiles, pulseTypes, stimDelays, ISIs,
         pulseNames, pulseSets, xsg, conditions) = lib.mat2py.getPulsesFromLegend(legendMat)
    elif os.path.exists(legendCSV):
        print('reading pulse metadata from pulseLegendQcam.csv')
        (pulseFiles, pulseTypes, stimDelays, ISIs,
         pulseNames, pulseSets, xsg, conditions) = lib.mat2py.getPulsesFromCSV(legendCSV)
    else:
        # No pulse metadata: write file inventory only, with NaN floats
        # (homogeneous column types so DynamicTable / NWBInspector are happy).
        print('no pulse metadata found — writing file inventory only')
        pulseFiles = list(qcam_files)
        pulseTypes = list(file_types)
        n = len(pulseFiles)
        stimDelays = [float('nan')] * n
        ISIs = [float('nan')] * n
        pulseNames = [''] * n
        pulseSets = [''] * n
        xsg = [''] * n
        conditions = [''] * n

    # cross-reference every pulse row to an acquisition (strict match)
    acq_idx, acq_starts, acq_nframes, acq_treatment = [], [], [], []
    for f in pulseFiles:
        try:
            i = qcam_files.index(f)
        except ValueError:
            raise ValueError(
                f'Pulse legend references {f!r}, which is not in the '
                f'qcamraw file list. Check {qcam_files}.'
            )
        acq_idx.append(f'OnePhotonSeries_{i:03}')
        acq_starts.append(float(starts_rel[i]))
        acq_nframes.append(int(nframes[i]))
        acq_treatment.append(treatments[i])

    stimData = {
        'file':            ('name of .qcamraw file', list(pulseFiles)),
        'OnePhotonSeries': ('OnePhotonSeries name', acq_idx),
        'starting_time':   ('start time of file (s) from session start',
                            acq_starts),
        'type':            ('whether stim or mapping type', list(pulseTypes)),
        'nFrames':         ('number of frames in file', acq_nframes),
        'frameRate':       ('frame rate of file',
                            [float(frame_rate)] * len(pulseFiles)),
        'treatment':       ('treatment', acq_treatment),
        'condition':       ('experimental condition', list(conditions)),
        'pulseNames':      ('sound stimulation pulse name', list(pulseNames)),
        'pulseSets':       ('sound stimulation pulse set', list(pulseSets)),
        'ISI':             ('ISI between pulses (s)', list(ISIs)),
        'stimDelay':       ('delay to start of pulses (s)', list(stimDelays)),
        'xsg':             ('associated .xsg file', list(xsg)),
    }
    cols = [VectorData(name=k, description=v[0], data=v[1])
            for k, v in stimData.items()]
    stim_table = DynamicTable(
        name='stim param table',
        description='Maps sound stim parameters to .qcamraw files',
        columns=cols,
    )
    nwbfile.add_stimulus(stim_table)
    print('added stim table data')