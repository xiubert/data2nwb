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
    helper for grabbing a list of strings from list of h5 references.
    """
    string_list = []
    for r in h5references:
        string_list.append("".join(chr(c.item()) for c in h5file[r][:]))
    return string_list


def getMatCellArrayOfStr(matPath: str, varPath: list[str]) -> list[str]:
    """
    Returns list of strings from cell array of strings in matlab.
    matPath: file path of .mat file
    varPath: nested path of cell array in mat file
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
    matPath: file path of .mat file
    varPath: nested path of cell array in mat file
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
    """
    moCorrData = loadmat(moCorrMatPath)
    # shifts = []
    # for cond in moCorrData['NoRMCorreParams'].dtype.fields:
    #     for shift in moCorrData['NoRMCorreParams'][cond][0][0]['shifts'][0][0]['shifts']:
    #         shifts.append(np.squeeze(shift[0]))
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



def getPupilData(pupilMat: str, experimentDir: str, getImgData: bool = False, pupilFrameRate: float = 10):
    """
    Helper to grab relevant pupil data from experiment dir.
    """
    getNestedStructData = lambda x: x[0][0]
    pupilMatData = loadmat(os.path.join(experimentDir,pupilMat))

    pupilData = {
        'pupilFrameFiles': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['pupilFrameFile'])),
        'pupilRadius': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['pupilRad'])),
        'DeepLabCutModel': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['model']))
    }
    pupilImgData = loadmat(os.path.join(experimentDir,pupilData['pupilFrameFiles'][0]))['pupilFrames']
    nFrames = [pupilImgData.shape[2]]
    for pupilFrameFile in pupilData['pupilFrameFiles'][1:]:
        pupilFrames = loadmat(os.path.join(experimentDir,pupilFrameFile))['pupilFrames']
        nFrames.append(pupilFrames.shape[2])
        if getImgData:
            pupilImgData = np.append(pupilImgData,
                    pupilFrames,
                    axis=2)
    if getImgData:
        pupilData['pupilImgData'] = np.transpose(pupilImgData, (2, 0, 1))
    pupilData['nFrames'] = nFrames
    pupilData['frameRate'] = [pupilFrameRate]*len(pupilData['nFrames'])

    return pupilData

# todo: split into getMetadata and get pupilImgData so that all of pupil data needn't be in memory.
def getPupilDataProcessed(pupilMatPath: str, pupilFrameRate: float = 10):
    """
    Helper to grab relevant pupil data from experiment dir.
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
    """
    pupilImgData = loadmat(pupilMatPath)['pupilFrames']

    return np.transpose(pupilImgData, (2, 0, 1))


def getPulses(tif: str, tifType: str) -> dict:
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
    """
    with h5py.File(tifFileListMatPath, "r") as h5:
        return list(h5['tifFileList'].keys())
    

def getTifList(dataPath: str, experimentID: str, tifListMatFilename: str):
    """
    Gets tif list, frame counts, and associated treatment from tifFileList.mat file.
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
    


