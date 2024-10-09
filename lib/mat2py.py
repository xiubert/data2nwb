"""
Helpers for pulling metadata from .mat files in ScanImage 2P experiment.
"""

import h5py
import numpy as np
from scipy.io import loadmat
import os

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

        my_string_list = []
        references = h5_ref[0]
        for r in references:
            my_string_list.append("".join(chr(c.item()) for c in h5[r][:]))
    return my_string_list


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
    

def getMoCorrShiftParams(moCorrMatPath: str) -> np.array:
    """
    Returns numpy array of xy translation shifts of raw .tif images.
    """
    moCorrData = loadmat(moCorrMatPath)
    shifts = []
    for cond in moCorrData['NoRMCorreParams'].dtype.fields:
        for shift in moCorrData['NoRMCorreParams'][cond][0][0]['shifts'][0][0]['shifts']:
            shifts.append(np.squeeze(shift[0]))
    params = moCorrData['NoRMCorreParams'][cond][0][0]['options_nonrigid'][0]

    return np.array(shifts),params


def getROImasks(roiMatPath: str) -> np.array:
    """
    Returns 1xNumberOfROI array of ROI image masks provided path to experiment .mat file containing
    ROIs drawn on motion corrected data.
    """
    roiData = loadmat(roiMatPath)
    return roiData['moCorROI'][0]['mask']


def getROIfluo(fluoMatPath: str, 
               varPath: list[str] = ['tifFileList','stim','moCorRawFroi']) -> np.array:
    """
    Extract fluorescence traces from tifFileList.
    Returns allFrames X ROI
    """
    with h5py.File(fluoMatPath, "r") as h5:
        h5_ref = h5
        for key in varPath:
            h5_ref = h5_ref[key]
        references = h5_ref[0]

        references = h5['tifFileList']['stim']['moCorRawFroi'][0]
        arr = np.array(h5[references[0]])
        
        for ref in references[1:]:
            arr = np.append(arr,np.array(h5[ref]),0)

    return arr



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



