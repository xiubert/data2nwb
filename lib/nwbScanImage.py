
import os
import glob
import re
import lib.mat2py
import lib.tifExtract
import numpy as np
from uuid import uuid4
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
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


"""
Functions for generating standard NWB files for ScanImage experiments.
"""

# common experiment params
PARAMS_nwbFile = {
    'lab': 'Tzounopoulos Lab',
    'institution': 'University of Pittsburgh',
}

PARAMS_nwbFilePC = {
    'experimenter': 'Patrick Cody',
    'lab': 'Tzounopoulos Lab',
    'institution': 'University of Pittsburgh',
    'related_publications': '10.1523/JNEUROSCI.0939-23.2024'
}

PARAMS_imagingPC = {
    'imaging_scopeDesc': (
        "Sutter moveable objective microscope (MOM) with mode-locked laser light (MaiTai HP) "
        "at 100-200 mW intensity through 40x0.8NA objective (Olympus) with X-Y galvanometric scanning"
    ),
    'imaging_manufacturer': 'Sutter',
    'imaging_opticalChannel0desc': "green channel for GCaMP",
    'imaging_emissionLambda': 525.0,
    'imagingPlane_rate': 5.0,
    'imagingPlane_desc': 'Auditory Cortex',
    'imagingPlane_excitationLambda': 940.0,
    'imagingPlane_loc': 'ACtx',
    'imagingPlane_indicator': 'GFP',
    'imagingPlane_gridSpacing': [145.0, 145.0],
    'imagingPlane_gridSpacingUnit': 'micrometers',
    'imagingPlane_originCoords': [-2.0, 4.25, 2.0],
    'imagingPlane_originCoordsUnit': 'meters'
}


def writeNWB(outputPath: str, nwbfile: NWBFile, overWrite: bool = True) -> bool:
    """
    Simple helper to write NWB file.
    """
    if os.path.exists(outputPath):
        if overWrite==True:
            os.remove(outputPath)
        else:
            raise('file exists')

    with NWBHDF5IO(outputPath, "w") as io:
        io.write(nwbfile)
    return True


def genNWBfromScanImage_pc(experimentID: str, dataPath: str, NWBoutputPath: str,
                            session_description: str,
                            experiment_description: str,
                            keywords: list[str],
                            experimenter: str,
                            related_publications: str,
                            lab: str,
                            institution: str,
                            imaging_scopeDesc: str,
                            imaging_manufacturer: str,
                            imaging_opticalChannel0desc: str,
                            imaging_emissionLambda: float,
                            imagingPlane_rate: str,
                            imagingPlane_desc: str,
                            imagingPlane_excitationLambda: float,
                            imagingPlane_loc: str,
                            imagingPlane_indicator: str,
                            imagingPlane_gridSpacing: list[float],
                            imagingPlane_gridSpacingUnit: str,
                            imagingPlane_originCoords: list[float],
                            imagingPlane_originCoordsUnit: str,
                            overWriteNWB: bool = True,
                            motionCorrectedTifDir: str = 'NoRMCorred',
                            returnNWB: bool = False
                           ) -> NWBFile:
    """
    A simple function to generate a standardized NWB file for a 
    given experiment directory for a given experiment ID.

    Session start assumed to be time of first .tif.
    """
    #%% set directories
    experimentDir = os.path.join(dataPath,experimentID)
    # experiment_mat = f"{experimentID}_anmlROI_stimTable.mat"
    moCorrMat = f"{experimentID}_NoRMCorreParams.mat"
    fluorescenceMat = f"{experimentID}_tifFileList.mat"

    pupilMat = f"{experimentID}_pulsePupilUVlegend2P_s.mat"

    #%% get tif file list
    # get tif creation date, end write time, and frame counts
    # tifFileList = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),
    #                                               varPath = ['tifFileList','stim','name'])
    # tifFrameCounts = lib.mat2py.getMatCellArrayOfNum(os.path.join(experimentDir,experiment_mat),
    #                                                  varPath = ['tifFileList','stim','nFrames'])
    # treatment = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),
    #                                             varPath = ['tifFileList','stim','treatment'])
    tifFileList, tifTypeList, tifFrameCounts, treatment = lib.mat2py.getTifList(
        dataPath,experimentID,f"{experimentID}_tifFileList.mat")

    
    # ROI mat ([experimentID]_moCorrROI_all.mat) will always end in _all.mat if treatment is 'none'.
    # If treatment (eg preZX1 and postZX1), there may be either one ROI mat for the whole session (ending in _all.mat) 
    # or an ROI .mat for each treatment (eg. [experimentID]_moCorrROI_preZX1.mat and [experimentID]_moCorrROI_postZX1.mat)

    # If treatment == 'none' (should be same for all .tif), then ROI mat will be called [experimentID]_moCorrROI_all.mat
    # and keys in imaging_plane and ImageSegmentation will be called 'all'

    # If multiple ROI sets / ROI change with time, must have different imaging_plane for each ROI set (via create_imaging_plane)

    # TwoPhotonSeries must be associated with the corresponding imaging plane
    # If multiple ROI .mat, there will be a separate ImageSegmentation (stored in plane_seg) for each,
    # this will reference the corresponding imaging plane.
    # ROI table regions are created for each ImageSegmentation (stored in plane_seg) and house
    # ROI fluorescence data.
    # Each RoiResponseSeries (eg fluorescence for a given .tif) is associated with the corresponding ROI table.
    roiMatPat = f"{experimentID}_moCorrROI*.mat"
    roiMats = glob.glob(os.path.join(experimentDir,roiMatPat))

    if len(roiMats)==1 and os.path.basename(roiMats[0])==f"{experimentID}_moCorrROI_all.mat":
        roiSet = ["all"]
        tifROI = roiSet*len(treatment)
    else:
        roiSet = [re.search(f"{experimentID}_moCorrROI_(.*).mat",roiMat).group(1) for roiMat in roiMats]
        tifROI = treatment

    # get metadata from first tif for session start
    session_start = lib.tifExtract.getSItifTime(os.path.join(experimentDir,tifFileList[0]))
    session_start

    print(list(zip(tifFileList,tifFrameCounts,treatment)))

    #%% NWB file generation
    # instantiate
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=session_start,
        experimenter=[
            experimenter,
        ],
        lab=lab,
        institution=institution,
        experiment_description=experiment_description,
        keywords=keywords,
        related_publications=related_publications,
        )
    
    #set imaging plane
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

    # if multiple ROI sets / ROI change with time, must have different imaging plane for each ROI set --> usually the case with pre/post treatment (ZX1)
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
    # “Origin coordinates are relative to bregma. First dimension corresponds to anterior-posterior axis (larger index = more anterior). 
    # Second dimension corresponds to medial-lateral axis (larger index = more rightward). 
    # Third dimension corresponds to dorsal-ventral axis (larger index = more ventral).

    # set 2p series data
    fileTimesInstantiate, nFrames, frameRates, starts = [],[],[],[]
    two_p_series = []
    for i,tif in enumerate(tifFileList):
        imgData,fileTimeInstantiate,frameCount,frameRate = lib.tifExtract.getSItifData(os.path.join(experimentDir,tif))
        start = lib.tifExtract.secMicroSec2sec(fileTimeInstantiate-session_start)
        two_p_ser = TwoPhotonSeries(
            name=f"TwoPhotonSeries_{i:03}",
            description="Raw 2P data",
            data=imgData,
            imaging_plane=imgPlane[tifROI[i]],
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
    shifts,moCorrParams = lib.mat2py.getMoCorrShiftParams(os.path.join(experimentDir,motionCorrectedTifDir,moCorrMat),nFrames=nFrames)

    # add processing module to include motion corrected data
    ophys_module = nwbfile.create_processing_module(
        name="ophys", description="optical physiology processed data"
    )
    motion_correction = MotionCorrection(name='Motion Corrected TwoPhotonSeries')

    for i,(tif,shift) in enumerate(zip(tifFileList,shifts)):
        imgData = lib.tifExtract.getSItifData(os.path.join(experimentDir,
                                                        motionCorrectedTifDir,
                                                        tif.replace('.tif','_NoRMCorre.tif')),getMetadata=False)

        corrected = ImageSeries(
            name="corrected",  # this must be named "corrected"
            description=f"A motion corrected image stack for acquisition {i:03}",
            data=imgData,
            unit="na",
            format="raw",
            comments=f"corrected file: {tif}",
            rate=frameRates[i],
            starting_time=starts[i],
        )

        xy_translation = TimeSeries(
            name=f"xy_translation",
            description=f"x,y translation in pixels for acquisition {i:03}",
            data=shift,
            unit="pixels",
            rate=frameRates[i],
            starting_time=starts[i],
            control_description = (moCorrParams if i==0 else None),
            comments=('control_description: NoRMCorreParams' if i==0 else ''),
        )

        motion_correction.add_corrected_image_stack(CorrectedImageStack(
            corrected=corrected,
            original=two_p_series[i],
            xy_translation=xy_translation,
            name=f"motion_corrected_TwoPhotonSeries_{i:03}"
        ))
    ophys_module.add(motion_correction)
    print("added motion correction data")


    # add ROI via planeSegmentation
    # - in experiment dir, ROI drawn on motion corrected data saved in [animal id]_moCorrROI_*.mat
    # - if no treatment file takes name [animal id]_moCorrROI_all.mat
    # - if more than one condition / treatment, image segmentation must be associated with a separate imaging plane
    img_seg = ImageSegmentation()
    ophys_module.add(img_seg)

    plane_seg = {}
    # usually roiMat for each treatment
    for roiMat,roiCond in zip(roiMats,roiSet):
        # treatCond = ('none' if roiCond=='all' else roiCond)
        roiMasks = lib.mat2py.getROImasks(roiMat)

        plane_seg[roiCond] = img_seg.create_plane_segmentation(
            name=f"PlaneSegmentation_{roiCond}",
            description=f"output from segmenting the imaging plane for {roiCond}",
                imaging_plane=imgPlane[roiCond],
                reference_images=[p for p,t in zip(two_p_series,tifROI) if t==roiCond],  # optional
            )

        for roiImageMask in roiMasks:
            # add image mask to plane segmentation
            plane_seg[roiCond].add_roi(image_mask=roiImageMask)
    print("added ROI segmentation data")

    # add fluorescence traces for ROIs
    # load ROI fluo data from experiment
    fluoROI = lib.mat2py.getROIfluo(os.path.join(experimentDir,fluorescenceMat))

    # roi fluorescence responses associated with a region, each region is associated with a plane segmentation (usually one per condition)
    # which has corresponding IDs for the ROI - roiResponseSeries a linked to these planesegment IDs
    roi_resp_series = []
    for cond in plane_seg:
        rt_region = plane_seg[cond].create_roi_table_region(
            region=plane_seg[cond].id.data, description=f"ROI for {cond}"
            )
        # only get responses in the matching treatment condition
        responses = [(i,f,s,fr) for i,(f,t,s,fr) in enumerate(zip(fluoROI,tifROI,starts,frameRates)) if t==cond]
        
        for i,f,s,fr in responses:
            roi_resp_series.append(RoiResponseSeries(
                name=f"RoiResponseSeries_{i:03}",
                description=f"Fluorescence responses for motion corrected ROIs for TwoPhotonSeries_{i:03}",
                data=f,
                rois=rt_region,
                unit="lumens",
                rate=fr,
                starting_time=s
                ))
    
    # one fluorescence module, RoiResponseSeries is a list
    fl = Fluorescence(roi_response_series=roi_resp_series)
    ophys_module.add(fl)
    print('added fluorescence trace data for ROIs')


    # add sound stimulus data via DynamicTable
    # pulseNames = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),['pulseLegend2P','pulseName'])
    # pulseSets = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),['pulseLegend2P','pulseSet'])
    # tifPulse = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),['pulseLegend2P','tif'])
    # pulseNames, pulseSets = zip(*[(pulseName,pulseSet) for t,pulseName,pulseSet in zip(tifPulse,pulseNames,pulseSets) if t in tifFileList])
    
    # stimData = {
    #     'file': ('name of .tif file',tifFileList),
    #     'fileTimeInstantiate': ('time .tif file was instantiated/created',
    #                             [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in fileTimesInstantiate]),
    #     'starting_time': ('starting time of .tif in seconds from first .tif',starts),
    #     'nFrames': ('number of frames in .tif file',nFrames),
    #     'frameRate': ('frame rate of .tif file',frameRates),
    #     'pulseNames': ('sound stimulation pulse name',pulseNames),
    #     'pulseSets': ('sound stimulation pulse set',pulseSets),
    #     'treatment': ('treatment',treatment)
    # }
    stimDelays, ISIs, pulseNames, pulseSets, xsg = lib.mat2py.getTifPulses(
        dataPath,experimentID,tifFileList,tifTypeList)
    
    stimData = {
    # 'fileTimeInstantiate': ('time .tif file was instantiated/created',
    #                         [dt.strftime('%d-%b-%Y %H:%M:%S.%f') for dt in fileTimesInstantiate]),
    'file': ('name of .tif file',tifFileList),
    'fileTimeInstantiate': ('time .tif file was instantiated/created',
                        [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in fileTimesInstantiate]),
    'starting_time': ('starting time of .tif in seconds from first .tif',starts),
    'type': ('whether stim or mapping type', tifTypeList),
    'nFrames': ('number of frames in .tif file',nFrames),
    'frameRate': ('frame rate of .tif file',frameRates),
    'treatment': ('treatment',treatment),
    'pulseNames': ('sound stimulation pulse name',pulseNames),
    'pulseSets': ('sound stimulation pulse set',pulseSets),
    'ISI': ('ISI between pulses in seconds', ISIs),
    'stimDelay': ('delay to start of pulses in seconds', stimDelays),
    'xsg': ('associated .xsg file storing raw pulse data',xsg)
    }
    
    cols = []
    for col,v in stimData.items():
        cols.append(
                VectorData(
                name=col,
                description=v[0],
                data=v[1],
            )
        )
    stim_table = DynamicTable(
        name='stim param table',
        description='Maps sound stim parameters to .tif files',
        columns=cols,
    )

    nwbfile.add_stimulus(stim_table)
    print('added stim table data')

    #%% add pupillometry
    if (os.path.exists(os.path.join(experimentDir,pupilMat)) or 
        os.path.exists(os.path.join(experimentDir,pupilMat.replace('_s.mat','.mat')))
        ):
        print(f"found pupillometry data for {experimentID}")

        behavior_module = nwbfile.create_processing_module(
                name="behavior", description="Processed behavioral data"
            )
        
        # add pupil radius split by tif file
        pupilDataProcessed = lib.mat2py.getPupilDataProcessed(os.path.join(experimentDir,pupilMat))

        # keep only those in tifList
        pupilTifIDs,pupilFrameFiles = zip(*[(i,t.replace('.tif','_pupilFrames.mat')) for i,t in enumerate(tifFileList) 
                                   if t.replace('.tif','_pupilFrames.mat') in pupilDataProcessed['pupilFrameFiles']])
        pupilRadii = [r for r,p in zip(pupilDataProcessed['pupilRadius'],pupilDataProcessed['pupilFrameFiles']) 
                    if p in pupilFrameFiles]
        
        # pupilFrameFiles,pupilRadius = zip(*[(f,r) for f,r in zip(pupilDataProcessed['pupilFrameFiles'],pupilDataProcessed['pupilRadius']) 
        #     if f.replace('_pupilFrames.mat','.tif') in tifFileList])

        pupil_radii = []

        for pupilTifID,pupilFrameFile,pupilRadius in zip(pupilTifIDs,pupilFrameFiles,pupilRadii):
            pupil_radii.append(
                TimeSeries(
                    name=f"pupil_radius_{pupilTifID:03}",
                    description=f"Pupil radius extracted from the video of the right eye for TwoPhotonSeries_{i:03}",
                    data=pupilRadius,
                    rate=float(pupilDataProcessed['frameRate']),
                    starting_time=starts[pupilTifID],
                    unit="na",
                    comments=f"pupilFrameFile: {pupilFrameFile}, associated .tif file: {tifFileList[pupilTifID]}"
                )
            )

        pupil_tracking = PupilTracking(time_series=pupil_radii, name="PupilTracking")
        behavior_module.add(pupil_tracking)
        print('added pupil radius data')
        
        # add pupil video
        pupilVideoSeries = []

        for pupilTifID,pupilFrameFile in zip(pupilTifIDs,pupilFrameFiles):
            pupilVideoSeries.append(
                ImageSeries(
                    name=f"pupil_video_{pupilTifID:03}",
                    description=f"Pupil video of the right eye for TwoPhotonSeries_{pupilTifID:03}",
                    data=lib.mat2py.getPupilImg(os.path.join(experimentDir,pupilFrameFile)),
                    rate=float(pupilDataProcessed['frameRate']),
                    starting_time=starts[pupilTifID],
                    unit="na",
                    comments=f"pupilFrameFile: {pupilFrameFile}, associated .tif file: {tifFileList[pupilTifID]}"
                )
            )
        behavior_module.add(pupilVideoSeries)
        print('added pupil video data')

    #%% write output
    writeNWB(NWBoutputPath,nwbfile,overWrite=overWriteNWB)

    print(f'NWB write success to: {NWBoutputPath}')
    if returnNWB==True:
        return nwbfile





