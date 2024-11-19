"""
Helpers for pulling metadata from .mat files in ScanImage 2P experiment.
"""

import h5py
import numpy as np
from scipy.io import loadmat
import os
import lib.tifExtract


def getH5stringList(h5references,h5file):
    """
    Helper for grabbing a list of strings from list of h5 references.

    Args:
        h5references (list[h5reference]): each element an h5 reference object
        h5file (h5file object)
    Returns:
        string_list (list[str]): each element a string from h5reference
    """
    string_list = []
    for r in h5references:
        string_list.append("".join(chr(c.item()) for c in h5file[r][:]))
    return string_list


def getMatCellArrayOfStr(matPath: str, varPath: list[str]) -> list[str]:
    """
    Returns list of strings from cell array of strings in matlab.

    Args:
        matPath (str): file path of .mat file
        varPath (list[str]): nested path of cell array in mat file
    Returns: 
        string_list (list[str]): each element a string from h5reference
    """
    with h5py.File(matPath, "r") as h5:
        h5_ref = h5
        for key in varPath:
            h5_ref = h5_ref[key]

        references = h5_ref[0]
        string_list = getH5stringList(references,h5)
    return string_list


def getMatCellArrayOfNum(matPath: str, varPath: list[str]) -> list[float]:
    """
    Returns list of nums from cell array of num in matlab.

    Args:
        matPath (str): file path of .mat file
        varPath (list[str]): nested path of cell array in mat file

    Returns:
        numList (list[float]): each element a float from h5reference
    """
    with h5py.File(matPath, "r") as h5:
        h5_ref = h5
        for key in varPath:
            h5_ref = h5_ref[key]

        numList = []
        references = h5_ref[0]
        for r in references:
            numList.append(np.array(h5[r])[0][0])
        return numList
    

def getMoCorrShiftParams(moCorrMatPath: str, nFrames: list[int] = None, concatenate: bool = False) -> np.ndarray:
    """
    Returns numpy array of xy translation shifts of raw .tif images.

    Args:
        moCorrMatPath (str): file path of motion correction parameters file eg AA0304_NoRMCorreParams.mat
        nFrames (list[int]): each element a frame count for associated .tif file
        concatenate (bool): whether or not to return shifts in one concatenated array

    Returns:
        allshifts (list): each element is shifts for associated .tif file
        params: motion correction parameters

    """
    moCorrData = loadmat(moCorrMatPath)

    for i,cond in enumerate(moCorrData['NoRMCorreParams'].dtype.fields):
        if i==0:
            # unnest
            shifts = np.stack(moCorrData['NoRMCorreParams'][cond][0][0]['shifts'][0][0]['shifts'].squeeze()).squeeze()
        else:
            shifts = np.append(shifts,np.stack(moCorrData['NoRMCorreParams'][cond][0][0]['shifts'][0][0]['shifts'].squeeze()).squeeze(),
                            axis=0)
            
    params = moCorrData['NoRMCorreParams'][cond][0][0]['options_nonrigid'][0]

    if concatenate==True:
        return shifts
    
    cumframes = np.cumsum(nFrames)
    firstShift = cumframes-nFrames
    lastShift = cumframes-1

    allshifts = []
    for start,end in zip(firstShift,lastShift):
        allshifts.append(shifts[start:end+1])

    return allshifts,params


def getROImasks(roiMatPath: str) -> np.ndarray:
    """
    Returns 1xNumberOfROI array of ROI image masks provided path to experiment .mat file containing
    ROIs drawn on motion corrected data.

    Args:
        roiMatPath (str): file path to .mat containing ROI segmentation data eg AA0304_moCorrROI_all.mat
    Returns:
        IDs (np.array): numpy array of int corresponding to index of ROI
        masks (np.array): numpy array of ROI masks
    """
    roiData = loadmat(roiMatPath)
    masks = roiData['moCorROI'][0]['mask']
    IDs = np.concatenate(roiData['moCorROI'][0]['ID']).astype(int)

    return IDs,masks


def getROIfluo(fluoMatPath: str, 
               concatenate: bool = False) -> tuple[list[str],list[np.ndarray]]:
    """
    Extract motion corrected fluorescence traces from tifFileList.
    Returns corresponding tif, allFrames X ROI

    Args:
        fluoMatPath (str): file path to .mat containing ROI fluorescence traces eg AA0304_tifFileList.mat
        concatenate (bool): whether or not to concatenate all responses across time/frames
    
    Returns:
        tifs (list):each element a tif file names associated with each response trace
        responses: list of numpy arrays of fluorescence responses for each ROI
    """
    with h5py.File(fluoMatPath, "r") as h5:
        tifTypes = list(h5['tifFileList'].keys())

        tifs,responses = [],[]
        for tifType in tifTypes:
            tifNameRefs = h5['tifFileList'][tifType]['name'][0]
            tifs.extend(lib.mat2py.getH5stringList(tifNameRefs,h5))
        
            responseRefs = h5['tifFileList'][tifType]['moCorRawFroi'][0]
            arr = (np.array(h5[responseRefs[0]]) if concatenate else [np.array(h5[responseRefs[0]])])
            for ref in responseRefs[1:]:
                if concatenate==True:
                    arr = np.append(arr,np.array(h5[ref]),0)
                else:
                    arr.append(np.array(h5[ref]))
            responses.extend(arr)

    return tifs,responses


def getPupilDataProcessed(pupilMatPath: str, pupilFrameRate: float = 10):
    """
    Helper to grab relevant pupil metadata from experiment dir. 

    Args:
        pupilMatPath (str): file path of .mat containing pupil data eg. AA0304_pulsePupilUVlegend2P_s.mat (pupil data stored as struct in .mat)
        pupilFrameRate (float): frame rate of pupil video
    Returns:
        pupilDataProcessed (dict): pupillometry metadata
    """
    getNestedStructData = lambda x: x[0][0]
    pupilMatData = loadmat(pupilMatPath)

    pupilDataProcessed = {
        'pupilFrameFiles': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['pupilFrameFile'])),
        'pupilRadius': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['pupilRad'])),
        'DeepLabCutModel': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['model'])),
        'frameRate': pupilFrameRate
    }

    return pupilDataProcessed


def getPupilImg(pupilMatPath: str) -> np.ndarray:
    """
    Helper to grab pupil img data from save .mat file containing img frames.

    Args:
        pupilMatPath (str): file path of .mat containing pupil data eg. AA0304_pulsePupilUVlegend2P_s.mat (pupil data stored as struct in .mat)
    Returns:
        np.array of pupil video data (time) x (X) x (Y)
    """
    pupilImgData = loadmat(pupilMatPath)['pupilFrames']

    return np.transpose(pupilImgData, (2, 0, 1))


def getPulses(tif: str, tifType: str) -> dict:
    """
    Grabs stimulation metadata associated with provided .tif.

    Args:
        tif (str): file path to .tif
        tifType (str): whether .tif is associated with single stim or with multiple stims/pulses (in case of BF mapping)
    Returns:
        pulseParams (dict): parameters of delivered pulse stimuli
        pulse (dict): metadata associated with pulse/stimuli eg pulse name etc
    """
    mat = loadmat(tif.replace('.tif','_Pulses.mat'))

    pulseParams,pulse = {},{}
    pulseParams['traceAcqTime'] = lib.tifExtract.parse_datetime_list(mat['params'][0]['acquisitionStartTime'][0][0])
    pulseParams['stimDelay'] = mat['params'][0]['stimDelay'][0][0][0]
    pulseParams['ISI'] = mat['params'][0]['ISI'][0][0][0]

    if tifType=='stim':
        pulse['pulseSet'] = mat['pulse'][0]['pulseset'][0][0]
        pulse['pulseName'] = mat['pulse'][0]['pulsename'][0][0]
        pulse['xsg'] = mat['pulse'][0]['curXSG'][0][0].split('\\')[-1]
    elif tifType=='map':
        pulse['pulseSet'] = np.concatenate(np.squeeze(mat['pulse'][0]['pulseset']))
        pulse['pulseName'] = np.concatenate(np.squeeze(mat['pulse'][0]['pulsename']))
        pulse['xsg'] = list(map(lambda x: x.split('\\')[-1],np.concatenate(np.squeeze(mat['pulse'][0]['curXSG']))))
    
    return pulseParams,pulse


def getTifTypes(tifFileListMatPath: str) -> list[str]:
    """
    Returns available tif types from tifFileList mat file.

    Args:
        tifFileListMatPath (str): filepath to .mat containing metadata of .tif files eg AA0304_tifFileList.mat
    Returns:
        list with each element indicated whether tif has single pulse/stimulus or multiple (in case of BF mapping .tif)
    """
    with h5py.File(tifFileListMatPath, "r") as h5:
        return list(h5['tifFileList'].keys())
    

def getTifList(dataPath: str, experimentID: str, tifListMatFilename: str):
    """
    Gets tif list, frame counts, and associated treatment from tifFileList.mat file.

    Args:
        dataPath (str): parent folder path to tifFileList.mat
        experimentID: experiment id (usually parent folder name eg AA0304)
        tifListMatFilename: filename of tifFileList.mat eg AA0304_tifFileList.mat
    Returns:
        tifList (list[str]): each element a .tif filename
        tifTypeList (list[str]): each element indicates .tif type (single pulse or multiple pulses/BFmap)
        treatments (list[str]): element indicates treatment associated w/ .tif
        nFrames (list[int]): element indicates frame count for associated .tif
    """
    tifListMatPath = os.path.join(dataPath,experimentID,tifListMatFilename)
    tifTypes = getTifTypes(tifListMatPath)

    tifList, tifTypeList, nFrames, treatments = [], [], [], []

    for tifType in tifTypes:
        print(f"recording includes: {tifType}")
        tifs = getMatCellArrayOfStr(tifListMatPath,varPath = ['tifFileList',tifType,'name'])
        treatment = getMatCellArrayOfStr(tifListMatPath,varPath = ['tifFileList',tifType,'treatment'])
        nFrame = getMatCellArrayOfNum(tifListMatPath,varPath = ['tifFileList',tifType,'nFrames'])
        if tifType=='map':
            treatment = [t.replace('BFmap','').strip() for t in treatment]
        tifList.extend(tifs)
        tifTypeList.extend([tifType]*len(tifs))
        nFrames.extend(nFrame)
        treatments.extend(treatment)
    
    return tifList, tifTypeList, treatments, nFrames


def getTifPulses(dataPath: str, experimentID: str, tifList: list[str], tifTypeList: list[str]):
    """
    Gets associated .tif pulse data.

    Args:
        dataPath (str): parent folder path to tifFileList.mat
        experimentID: experiment id (usually parent folder name eg AA0304)
        tifList (list[str]): each element is .tif filename
        tifTypeList (list[str]): each element indicates .tif type (single pulse or multiple pulses/BFmap)

    Returns:
        tifs (list[str]): each element is .tif filename
        tifTypes (list[str]): each element indicates .tif type (single pulse or multiple pulses/BFmap)
        stimDelays (list[int]): each element indicates delay to stimulus for associated .tif
        ISIs (list[int]): each element indicates ISI between pulses for assocaited .tif
        pulseNames (list[str]): each element is list of pulses for associated .tif
        pulseSets (list[str]): each element indicates pulse set name for associated .tif
        xsg (list[str]): each element indicates associated .xsg file(s) for assocaited .tif
    """
    tifs, tifTypes = [],[]
    stimDelays, ISIs, pulseNames, pulseSets, xsg = [],[],[],[],[]

    for tif,tifType in zip(tifList,tifTypeList):
        pulseParams,pulse = getPulses(os.path.join(dataPath,experimentID,tif),tifType)
        pulseSet,pulseName,x = list(pulse.values())

        if tifType=='map':
            tif,tifType,ISI,stimDelay = zip(*[(tif,
                                   tifType,
                                   pulseParams['stimDelay'],
                                   pulseParams['ISI']) for _ in pulseName])
            xsg.extend(x)
            pulseNames.extend(pulseName)
            pulseSets.extend(pulseSet)
            ISIs.extend(ISI)
            stimDelays.extend(stimDelay)
            tifs.extend(tif)
            tifTypes.extend(tifType)

        else:
            tifs.append(tif)
            tifTypes.append(tifType)
            xsg.append(x)
            pulseNames.append(pulseName)
            pulseSets.append((np.unique(pulseSet)[0] if len(np.unique(pulseSet))==1 else pulseSet))
            ISIs.append(pulseParams['ISI'])
            stimDelays.append(pulseParams['stimDelay'])
        
    return tifs, tifTypes, stimDelays, ISIs, pulseNames, pulseSets, xsg
    


