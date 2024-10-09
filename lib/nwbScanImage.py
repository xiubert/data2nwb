
import os
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


def writeNWB(outputPath: str, nwbfile, overWrite: bool = True):
    if os.path.exists(outputPath):
        if overWrite==True:
            os.remove(outputPath)
        else:
            raise('file exists')

    with NWBHDF5IO(outputPath, "w") as io:
        io.write(nwbfile)

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
                           ):
    """
    A simple function to generate a standardized NWB file for a 
    given experiment directory for a given experiment ID.

    Session start assumed to be time of first .tif.
    """
    #%% set directories
    # dataPath = "/home/pac/Documents/Python/nwb/scanimage/rawData"
    # experimentID = "AA0304"
    # NWBoutputPath = f"/home/pac/Documents/Python/nwb/scanimage/nwbOutput/{experimentID}.nwb"

    experimentDir = os.path.join(dataPath,experimentID)
    experiment_mat = f"{experimentID}_anmlROI_stimTable.mat"
    moCorrMat = f"./NoRMCorred/{experimentID}_NoRMCorreParams.mat"
    roiMat = f"{experimentID}_moCorrROI_all.mat"
    fluorescenceMat = f"{experimentID}_tifFileList.mat"

    pupilMat = f"{experimentID}_pulsePupilUVlegend2P_s.mat"

    #%% get tif file list
    # get tif creation date, end write time, and frame counts
    tifFileList = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),varPath = ['tifFileList','stim','name'])
    fileTimeWrite = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),varPath = ['tifFileList','stim','date'])
    tifFrameCounts = lib.mat2py.getMatCellArrayOfNum(os.path.join(experimentDir,experiment_mat),varPath = ['tifFileList','stim','nFrames'])
    treatment = lib.mat2py.getMatCellArrayOfNum(os.path.join(experimentDir,experiment_mat),varPath = ['tifFileList','stim','treatment'])
    print(list(zip(tifFileList,fileTimeWrite,tifFrameCounts)))

    #get tif data
    imgData,fileTimeInstantiate,nFrames,fr = lib.tifExtract.getTifData(tifFileList, experimentDir)
    print(f"total frames: {sum(nFrames)}")

    # Convert tif file instantiation date strings to timestamps of each frame in seconds
    timestamps,date_times = lib.tifExtract.filetime2secTimestamp(fileTimeInstantiate, nFrames, fr)
    print(f"length of timestamps: {len(timestamps)}")

    #%% NWB file generation
    # instantiate
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=date_times[0],
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

    imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane",
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
    two_p_series = TwoPhotonSeries(
        name="TwoPhotonSeries",
        description="Raw 2p data",
        data=imgData,
        imaging_plane=imaging_plane,
        unit="normalized amplitude",
        control_description=list(zip(fileTimeInstantiate,tifFileList,nFrames,fr,fileTimeWrite,treatment)),
        comments="control_description form: (fileTimeInstantiate,file,nFrames,frameRate,fileTimeWrite,treatment)",
        timestamps=timestamps
        )
    
    nwbfile.add_acquisition(two_p_series)
    print("added 2P data")

    # add motion correction data
    imgData_corr = lib.tifExtract.getTifData(
        [f.replace('.tif','_NoRMCorre.tif') for f in tifFileList], 
        os.path.join(experimentDir,'NoRMCorred'),
        getMetadata=False)

    shifts,moCorrParams = lib.mat2py.getMoCorrShiftParams(os.path.join(experimentDir,moCorrMat))

    corrected = ImageSeries(
        name="corrected",  # this must be named "corrected"
        description="A motion corrected image stack",
        data=imgData_corr,
        unit="na",
        format="raw",
        control_description=list(zip(fileTimeInstantiate,tifFileList,nFrames,fr,fileTimeWrite)),
        comments="control_description form: (fileTimeInstantiate,file,nFrames,frameRate,fileTimeWrite)",
        timestamps=timestamps
        )

    xy_translation = TimeSeries(
            name="xy_translation",
            description="x,y translation in pixels",
            data=shifts,
            unit="pixels",
            timestamps=timestamps,
            control_description = moCorrParams,
            comments= 'control_description: NoRMCorreParams',
        )

    corrected_image_stack = CorrectedImageStack(
            corrected=corrected,
            original=two_p_series,
            xy_translation=xy_translation,
        )

    motion_correction = MotionCorrection(corrected_image_stacks=[corrected_image_stack])

    ophys_module = nwbfile.create_processing_module(
        name="ophys", description="optical physiology processed data"
        )

    ophys_module.add(motion_correction)
    print("added motion correction data")

    # add ROI via planeSegmentation
    roiMasks = lib.mat2py.getROImasks(os.path.join(experimentDir,roiMat))

    img_seg = ImageSegmentation()

    ps = img_seg.create_plane_segmentation(
        name="PlaneSegmentation",
        description="output from segmenting the imaging plane",
        imaging_plane=imaging_plane,
        reference_images=two_p_series,  # optional
    )

    ophys_module.add(img_seg)

    for roiImageMask in roiMasks:
        # add image mask to plane segmentation
        ps.add_roi(image_mask=roiImageMask)
    print("added ROI segmentation data")

    # add fluorescence traces for ROIs
    # load ROI fluo data from experiment
    arrL = lib.mat2py.getROIfluo(os.path.join(experimentDir,fluorescenceMat))

    rt_region = ps.create_roi_table_region(
        region=list(range(arrL.shape[1])), description="all ROI"
        )

    roi_resp_series = RoiResponseSeries(
        name="RoiResponseSeries",
        description="Fluorescence responses for motion corrected ROIs",
        data=arrL,
        rois=rt_region,
        unit="lumens",
        rate=imagingPlane_rate,
    )

    fl = Fluorescence(roi_response_series=roi_resp_series)
    ophys_module.add(fl)
    print('added fluorescence trace data for ROIs')

    # add sound stimulus data via DynamicTable
    pulseNames = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),['pulseLegend2P','pulseName'])
    pulseSets = lib.mat2py.getMatCellArrayOfStr(os.path.join(experimentDir,experiment_mat),['pulseLegend2P','pulseSet'])

    stimData = {
        'fileTimeInstantiate': ('time .tif file was instantiated/created',fileTimeInstantiate),
        'file': ('name of .tif file',tifFileList),
        'nFrames': ('number of frames in .tif file',nFrames),
        'frameRate': ('frame rate of .tif file',fr),
        'pulseNames': ('sound stimulation pulse name',pulseNames),
        'pulseSets': ('sound stimulation pulse set',pulseSets)
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
        pupilData = lib.mat2py.getPupilData(pupilMat,getImgData=True,experimentDir=experimentDir)
        # if not adding pupil video to NWB
        # pupilData = lib.mat2py.getPupilData(pupilMat,getImgData=False,experimentDir=experimentDir)
    
        # get pupil time stamps
        pupilTimestamps,pupilDate_times = lib.tifExtract.filetime2secTimestamp(
            fileTimeInstantiate,pupilData['nFrames'],pupilData['frameRate'])

        # add pupil radius
        behavior_module = nwbfile.create_processing_module(
                name="behavior", description="Processed behavioral data"
            )

        pupil_diameter = TimeSeries(
                name="pupil_radius",
                description="Pupil radius extracted from the video of the right eye.",
                data=np.concatenate(pupilData['pupilRadius']),
                timestamps=pupilTimestamps,
                unit="na",
            )

        pupil_tracking = PupilTracking(time_series=pupil_diameter, name="PupilTracking")

        behavior_module.add(pupil_tracking)
        print('added pupil radius data')
        
        # add pupil video
        pupil_video = ImageSeries(
            name="pupil_video",
            description="Pupil video of the right eye.",
            data=pupilData['pupilImgData'],
            timestamps=pupilTimestamps,
            unit="arbitrary",
            )

        pupil_video_module = nwbfile.create_processing_module(
            name="pupillometry video", description="pupillometry video data"
        )

        pupil_video_module.add(pupil_video)
        print('added pupil video data')

    #%% write output
    writeNWB(NWBoutputPath,nwbfile,overWrite=overWriteNWB)

    print(f'NWB write success to: {NWBoutputPath}')





