import os
import glob
import re
import lib.mat2py
import lib.tifExtract
import numpy as np
from uuid import uuid4
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject
from pynwb.image import ImageSeries
from pynwb.ophys import (
    CorrectedImageStack,
    Fluorescence,
    ImageSegmentation,
    MotionCorrection,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)
from pynwb.behavior import PupilTracking
from hdmf.common import VectorData, DynamicTable
from hdmf.backends.hdf5.h5_utils import H5DataIO

"""
Functions for generating standard NWB files for ScanImage experiments.
"""

def setSubject(subject_id: str, age: str, species: str,
               sex: str, genotype: str,
               description: str):
    """
    Simple helper function to set experiment subject.

    Args:
        subject_id (str): id of animal subject (usually in format [A-Z]{2}/d{4} eg. AA0304)
        age: ISO 8601 Duration format, e.g., "P90D" for 90 days old
        species (str): The formal Latin binomial nomenclature, e.g., "Mus musculus", "Homo sapiens"
        sex (str): Single letter abbreviation, e.g., "F" (female), "M" (male), "U" (unknown), and "O" (other)
        genotype (str): genotype of subject eg. C57BL6/J or ZnT3KO
        description (str): informative description of subject

    Returns:
        nwb Subject object
    """
    subject = Subject(
        subject_id=subject_id,
        age=age,
        species=species,
        sex=sex,
        genotype=genotype,
        description=description,
    )

    return subject


def writeNWB(outputPath: str, nwbfile: NWBFile, overWrite: bool = True) -> bool:
    """
    Simple helper to write NWB file.

    Args:
        outputPath (str):  File path for desired NWB output file.
        nwbfile: NWB file object to be written to disk.
        overWrite (bool): Boolean to indicate whether or not
                          file should be overwriten if it exists

    Returns:
        bool: True if completed without error.
    """
    if os.path.exists(outputPath):
        if overWrite is True:
            os.remove(outputPath)
        else:
            raise FileExistsError(f'{outputPath} exists and overWrite=False')

    with NWBHDF5IO(outputPath, "w") as io:
        io.write(nwbfile)
    return True


def genNWBfromScanImage_pc(experimentID: str, dataPath: str, NWBoutputPath: str,
                           subject: Subject,
                           session_description: str,
                           experiment_description: str,
                           keywords: list,
                           experimenter: str,
                           related_publications: str,
                           lab: str,
                           institution: str,
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
                           overWriteNWB: bool = True,
                           motionCorrectedTifDir: str = 'NoRMCorred',
                           returnNWB: bool = False
                           ) -> NWBFile:
    """
    A simple function to generate a standardized NWB file for a
    given experiment directory (directory name assumed to match experiment ID).
    Saves NWB file to NWBoutputPath. Outputs NWB object if returnNWB True.

    Adds ophys, motion correction, ROI segmentation and traces, and stimulus metadata.
    Adds pupillometry if available.

    Session start assumed to be time of first recorded .tif (with lowest file number id, eg AA0304_0001.tif).
    """
    #%% set directories
    experimentDir = os.path.join(dataPath, experimentID)
    moCorrMat = f"{experimentID}_NoRMCorreParams.mat"
    fluorescenceMat = f"{experimentID}_tifFileList.mat"

    pupilMat = f"{experimentID}_pulsePupilUVlegend2P_s.mat"

    print(f"Generating standardized NWB file for {experimentID}")

    #%% get tif file list
    tifFileList, tifTypeList, treatment, _ = lib.mat2py.getTifList(
        dataPath, experimentID, fluorescenceMat)

    # ROI mat ([experimentID]_moCorrROI_all.mat) will always end in _all.mat if treatment is 'none'.
    # If treatment (eg preZX1 and postZX1), there may be either one ROI mat for the whole session
    # (ending in _all.mat) or an ROI .mat for each treatment (eg. [experimentID]_moCorrROI_preZX1.mat
    # and [experimentID]_moCorrROI_postZX1.mat)
    roiMatPat = f"{experimentID}_moCorrROI*.mat"
    roiMats = glob.glob(os.path.join(experimentDir, roiMatPat))

    if not roiMats:
        roiOutputMats = sorted(glob.glob(os.path.join(experimentDir, '*_roiOutput.mat')))
        if not roiOutputMats:
            raise FileNotFoundError(f"No ROI .mat files found in {experimentDir}")
        roiMats = [lib.mat2py._select_one(
            title='ROI file selection',
            prompt='Select roiOutput.mat file:',
            items=[os.path.basename(p) for p in roiOutputMats],
            base_dir=experimentDir
        )]

    if len(roiMats) == 1 and os.path.basename(roiMats[0]) == f"{experimentID}_moCorrROI_all.mat":
        roiSet = ["all"]
        tifROIset = roiSet * len(treatment)
    elif len(roiMats) == 1 and os.path.basename(roiMats[0]).endswith('_roiOutput.mat'):
        roiSet = ["all"]
        tifROIset = roiSet * len(treatment)
    else:
        roiSet = [re.search(f"{experimentID}_moCorrROI_(.*).mat", roiMat).group(1) for roiMat in roiMats]
        tifROIset = treatment

    # get metadata from first tif for session start
    session_start = lib.tifExtract.getSItifTime(os.path.join(experimentDir, tifFileList[0]))

    #%% NWB file generation
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

    # set imaging plane
    device = nwbfile.create_device(
        name="Microscope",
        description=imaging_scopeDesc,
        manufacturer=imaging_manufacturer,
    )
    optical_channel = OpticalChannel(
        name="OpticalChannel",
        description=imaging_opticalChannel0desc,
        emission_lambda=imaging_emissionLambda,
    )
    # filter FF03 525/50 (Semrock) --> 525 +/- 25 (center wavelength 525):
    # https://www.idex-hs.com/store/product-detail/ff03_525_50_25/fl-004656

    # if multiple ROI sets / ROI change with time, must have different imaging plane for each ROI set
    imgPlane = {}
    for cond in roiSet:
        imgPlane[cond] = nwbfile.create_imaging_plane(
            name=f"ImagingPlane_{cond}",
            optical_channel=optical_channel,
            imaging_rate=imagingPlane_rate,
            description=imagingPlane_desc,
            device=device,
            excitation_lambda=imagingPlane_excitationLambda,
            indicator=imagingPlane_indicator,
            location=imagingPlane_loc,
            grid_spacing=imagingPlane_gridSpacing,
            grid_spacing_unit=imagingPlane_gridSpacingUnit,
            origin_coords=imagingPlane_originCoords,
            origin_coords_unit=imagingPlane_originCoordsUnit,
        )

    # set 2p series data
    fileTimesInstantiate, nFrames, frameRates, starts = [], [], [], []
    two_p_series = []
    for i, tif in enumerate(tifFileList):
        imgData, fileTimeInstantiate, frameCount, frameRate = \
            lib.tifExtract.getSItifData(os.path.join(experimentDir, tif))
        start = lib.tifExtract.secMicroSec2sec(fileTimeInstantiate - session_start)
        two_p_ser = TwoPhotonSeries(
            name=f"TwoPhotonSeries_{i:03}",
            description="Raw 2P data",
            data=H5DataIO(data=imgData, compression=True),
            imaging_plane=imgPlane[tifROIset[i]],
            rate=frameRate,
            starting_time=start,
            unit="normalized amplitude",
            comments=f"file: {tif}, treatment: {treatment[i]}, fileTimeInstantiate: {fileTimeInstantiate}, nFrames: {frameCount})",
        )
        two_p_series.append(two_p_ser)

        fileTimesInstantiate.append(fileTimeInstantiate)
        nFrames.append(frameCount)
        frameRates.append(frameRate)
        starts.append(start)

        nwbfile.add_acquisition(two_p_ser)
    print("added 2P data")

    # add motion correction data
    shifts, moCorrParams = lib.mat2py.getMoCorrShiftParams(
        os.path.join(experimentDir, motionCorrectedTifDir, moCorrMat),
        nFrames=nFrames)

    # ophys processing module
    ophys_module = nwbfile.create_processing_module(
        name="ophys", description="optical physiology processed data"
    )
    motion_correction = MotionCorrection(name='Motion Corrected TwoPhotonSeries')

    for i, (tif, shift) in enumerate(zip(tifFileList, shifts)):
        imgData = lib.tifExtract.getSItifData(
            os.path.join(experimentDir, motionCorrectedTifDir,
                         tif.replace('.tif', '_NoRMCorre.tif')),
            getMetadata=False)

        corrected = ImageSeries(
            name="corrected",  # this must be named "corrected"
            description=f"A motion corrected image stack for acquisition {i:03}",
            data=H5DataIO(data=imgData, compression=True),
            unit="na",
            format="raw",
            comments=f"corrected file: {tif}",
            rate=frameRates[i],
            starting_time=starts[i],
        )

        xy_translation = TimeSeries(
            name="xy_translation",
            description=f"x,y translation in pixels for acquisition {i:03}",
            data=shift,
            unit="pixels",
            rate=frameRates[i],
            starting_time=starts[i],
            control_description=(moCorrParams if i == 0 else None),
            comments=('control_description: NoRMCorreParams' if i == 0 else ''),
        )

        motion_correction.add_corrected_image_stack(CorrectedImageStack(
            corrected=corrected,
            original=two_p_series[i],
            xy_translation=xy_translation,
            name=f"motion_corrected_TwoPhotonSeries_{i:03}"
        ))
    ophys_module.add(motion_correction)
    print("added motion correction data")

    # ---- ROI segmentation + fluorescence container ----
    # Both ImageSegmentation and Fluorescence are attached to the ophys
    # processing module BEFORE any RoiResponseSeries / DynamicTableRegion is
    # constructed, so HDMF's ancestor-chain validation passes cleanly.
    img_seg = ImageSegmentation()
    ophys_module.add(img_seg)

    fl = Fluorescence()
    ophys_module.add(fl)

    plane_seg = {}
    roiMasksPerCond = {}
    for roiMat, roiCond in zip(roiMats, roiSet):
        roiIDs, roiMasks = lib.mat2py.getROImasks(roiMat)
        roiMasksPerCond[roiCond] = roiMasks

        plane_seg[roiCond] = img_seg.create_plane_segmentation(
            name=f"PlaneSegmentation_{roiCond}",
            description=f"output from segmenting the imaging plane for {roiCond}",
            imaging_plane=imgPlane[roiCond],
            reference_images=[p for p, t in zip(two_p_series, tifROIset) if t == roiCond],
        )

        for roiID, roiImageMask in zip(roiIDs, roiMasks):
            plane_seg[roiCond].add_roi(id=roiID, image_mask=roiImageMask)
    print("added ROI segmentation data")

    # add fluorescence traces for ROIs
    fluoMatPath = os.path.join(experimentDir, fluorescenceMat)
    if os.path.exists(fluoMatPath):
        fluoTif, fluoROI = lib.mat2py.getROIfluo(fluoMatPath)
        fluoROI = [fluoROI[i] for i in np.where(np.isin(fluoTif, tifFileList))[0]]
    else:
        print("tifFileList not found — computing ROI fluorescence from motion-corrected tifs...")
        combined_masks = np.concatenate([roiMasksPerCond[c] for c in roiSet], axis=0)
        fluoTif, fluoROI = lib.mat2py.getROIfluoFromTifs(
            tifList=tifFileList,
            masks=combined_masks,
            tifDir=os.path.join(experimentDir, motionCorrectedTifDir),
        )

    # roi fluorescence responses associated with a region; each region linked to a plane segmentation
    for cond in plane_seg:
        rt_region = plane_seg[cond].create_roi_table_region(
            region=list(range(len(plane_seg[cond].id.data))),
            description=f"ROI for {cond}",
        )
        # only get responses in the matching treatment condition
        responses = [(i, fluo, start, fr)
                     for i, (fluo, titfSet, start, fr) in
                     enumerate(zip(fluoROI, tifROIset, starts, frameRates))
                     if titfSet == cond]
        for i, fluo, start, fr in responses:
            fl.create_roi_response_series(
                name=f"RoiResponseSeries_{i:03}",
                description=f"Fluorescence responses for motion corrected ROIs for TwoPhotonSeries_{i:03}",
                data=fluo,
                rois=rt_region,
                unit="lumens",
                rate=fr,
                starting_time=start,
            )
    print('added fluorescence trace data for ROIs')

    # add sound stimulus data via DynamicTable
    pulsesMatFiles = glob.glob(os.path.join(experimentDir, '*_Pulses.mat'))
    legendMat = os.path.join(experimentDir, 'pulseLegend2P.mat')
    legendCSV = os.path.join(experimentDir, 'pulseLegend2P.csv')
    if pulsesMatFiles:
        (pulseTifs, pulseTifTypes, stimDelays, ISIs,
         pulseNames, pulseSets, xsg, conditions) = lib.mat2py.getTifPulses(
            dataPath, experimentID, tifFileList, tifTypeList)
    elif os.path.exists(legendMat):
        print("No _Pulses.mat files found — reading pulse metadata from pulseLegend2P.mat...")
        (pulseTifs, pulseTifTypes, stimDelays, ISIs,
         pulseNames, pulseSets, xsg, conditions) = lib.mat2py.getPulsesFromLegend(legendMat)
    elif os.path.exists(legendCSV):
        print("No _Pulses.mat or pulseLegend2P.mat found — reading pulse metadata from pulseLegend2P.csv...")
        (pulseTifs, pulseTifTypes, stimDelays, ISIs,
         pulseNames, pulseSets, xsg, conditions) = lib.mat2py.getPulsesFromCSV(legendCSV)
    else:
        raise FileNotFoundError(
            f"No _Pulses.mat, pulseLegend2P.mat, or pulseLegend2P.csv found in {experimentDir}"
        )

    # extend remaining params
    pulseTwoPidx, pulseFileTimesInstantiatePulse, pulseStarts, pulseNframes = [], [], [], []
    pulseFrameRates, pulseTreatment, pulseConditions = [], [], []
    for tif, cond in zip(pulseTifs, conditions):
        tifIDX = tifFileList.index(tif)
        pulseTwoPidx.append(f"TwoPhotonSeries_{tifIDX:03}")
        pulseFileTimesInstantiatePulse.append(
            fileTimesInstantiate[tifIDX].strftime('%Y-%m-%d %H:%M:%S.%f'))
        pulseStarts.append(starts[tifIDX])
        pulseNframes.append(nFrames[tifIDX])
        pulseFrameRates.append(frameRates[tifIDX])
        pulseTreatment.append(treatment[tifIDX])
        pulseConditions.append(cond)

    stimData = {
        'file': ('name of .tif file', pulseTifs),
        'TwoPhotonSeries': ('TwoPhotonSeries index', pulseTwoPidx),
        'fileTimeInstantiate': ('time .tif file was instantiated/created',
                                pulseFileTimesInstantiatePulse),
        'starting_time': ('starting time of .tif in seconds from first .tif', pulseStarts),
        'type': ('whether stim or mapping type', pulseTifTypes),
        'nFrames': ('number of frames in .tif file', pulseNframes),
        'frameRate': ('frame rate of .tif file', pulseFrameRates),
        'treatment': ('treatment', pulseTreatment),
        'condition': ('experimental condition', pulseConditions),
        'pulseNames': ('sound stimulation pulse name', pulseNames),
        'pulseSets': ('sound stimulation pulse set', pulseSets),
        'ISI': ('ISI between pulses in seconds', ISIs),
        'stimDelay': ('delay to start of pulses in seconds', stimDelays),
        'xsg': ('associated .xsg file storing raw pulse data', xsg)
    }

    cols = []
    for col, v in stimData.items():
        cols.append(VectorData(name=col, description=v[0], data=v[1]))
    stim_table = DynamicTable(
        name='stim param table',
        description='Maps sound stim parameters to .tif files',
        columns=cols,
    )

    nwbfile.add_stimulus(stim_table)
    print('added stim table data')

    #%% add pupillometry
    if (os.path.exists(os.path.join(experimentDir, pupilMat)) or
        os.path.exists(os.path.join(experimentDir, pupilMat.replace('_s.mat', '.mat')))):
        print(f"found pupillometry data for {experimentID}")

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Processed behavioral data"
        )

        # ---- pupil radius split by tif file ----
        # PupilTracking is created empty and attached to the behavior module
        # before any of its child TimeSeries are constructed, matching the
        # ordering used for Fluorescence above.
        pupilDataProcessed = lib.mat2py.getPupilDataProcessed(os.path.join(experimentDir, pupilMat))

        # keep only those in tifList
        pupilTifIDs, pupilFrameFiles = zip(*[
            (i, t.replace('.tif', '_pupilFrames.mat'))
            for i, t in enumerate(tifFileList)
            if t.replace('.tif', '_pupilFrames.mat') in pupilDataProcessed['pupilFrameFiles']
        ])
        pupilRadii = [r for r, p in zip(pupilDataProcessed['pupilRadius'],
                                        pupilDataProcessed['pupilFrameFiles'])
                      if p in pupilFrameFiles]

        pupil_tracking = PupilTracking(name="PupilTracking")
        behavior_module.add(pupil_tracking)

        for pupilTifID, pupilFrameFile, pupilRadius in zip(pupilTifIDs, pupilFrameFiles, pupilRadii):
            pupil_tracking.create_timeseries(
                name=f"pupil_radius_{pupilTifID:03}",
                description=f"Pupil radius extracted from the video of the right eye for TwoPhotonSeries_{pupilTifID:03}",
                data=pupilRadius,
                rate=float(pupilDataProcessed['frameRate']),
                starting_time=starts[pupilTifID],
                unit="na",
                comments=f"pupilFrameFile: {pupilFrameFile}, associated .tif file: {tifFileList[pupilTifID]}",
            )
        print('added pupil radius data')

        # ---- pupil video ----
        # Each ImageSeries is added individually to the behavior module
        # (ProcessingModule.add expects a single container, not a list).
        for pupilTifID, pupilFrameFile in zip(pupilTifIDs, pupilFrameFiles):
            pupil_video = ImageSeries(
                name=f"pupil_video_{pupilTifID:03}",
                description=f"Pupil video of the right eye for TwoPhotonSeries_{pupilTifID:03}",
                data=H5DataIO(
                    data=lib.mat2py.getPupilImg(os.path.join(experimentDir, pupilFrameFile)),
                    compression=True),
                rate=float(pupilDataProcessed['frameRate']),
                starting_time=starts[pupilTifID],
                unit="na",
                comments=f"pupilFrameFile: {pupilFrameFile}, associated .tif file: {tifFileList[pupilTifID]}",
            )
            behavior_module.add(pupil_video)
        print('added pupil video data')

    #%% write output
    writeNWB(NWBoutputPath, nwbfile, overWrite=overWriteNWB)

    print(f'NWB write success to: {NWBoutputPath}')
    if returnNWB is True:
        return nwbfile