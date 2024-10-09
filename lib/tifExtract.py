"""
Helpers for grabbing image data and metadata from ScanImage .tif files
"""
import os
from datetime import datetime
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import (
            extract_extra_metadata,
            parse_metadata,
        )

def getTifData(tifFileList: list[str], tifDirectory: str, getMetadata: bool = True):
    """
    Grabs .tif image data as well as frame rate, file instantiation time (creation time), 
    and frame number from metadata.

    tifFileList: list of .tif files
    tifDirectory: path to directory containing .tif files
    """
    if getMetadata:
        fr = []
        fileTimeInstantiate = []
        nFrames = []
    for i,tif in enumerate(tifFileList):
        if i==0:
            imgData = ScanImageTiffReader(os.path.join(tifDirectory,tif)).data()
        else:
            imgData = np.append(imgData,ScanImageTiffReader(os.path.join(tifDirectory,tif)).data(),0)
        
        if getMetadata:
            image_metadata = extract_extra_metadata(file_path=os.path.join(tifDirectory,tif))
            fileTimeInstantiate.append(image_metadata['epoch'])
            extraMeta = parse_metadata(image_metadata)
            fr.append(extraMeta['sampling_frequency'])
            nFrames.append(extraMeta['frames_per_slice'])

    print(np.shape(imgData))
    if getMetadata:
        return imgData,fileTimeInstantiate,nFrames,fr
    return imgData


def parse_datetime_string(date_str: str) -> datetime:
    """
    Parse date string from ScanImage .tif metadata ('epoch') and convert it to a datetime object.
    """

    # Remove the brackets and split the string by commas
    date_str = date_str.strip('[]')
    date_list = [float(x) for x in date_str.split(',')]
    
    # Convert the list into a datetime object
    dt = datetime(
        year=int(date_list[0]),
        month=int(date_list[1]),
        day=int(date_list[2]),
        hour=int(date_list[3]),
        minute=int(date_list[4]),
        second=int(date_list[5]),  # Whole seconds part
        microsecond=int((date_list[5] % 1) * 1_000_000)  # Fractional seconds to microseconds
    )
    return dt


def filetime2secTimestamp(fileInstantiateTime: list[str], frameCounts: list[int], frameRates: list[float]):
    """
    Converts file instantiation times (from Scanimage .tif metadata ['epoch']) 
    to successive timestamps for all frames in seconds 
    using frame count and frame rate associated with each file date.
    """
    # Convert all date-time strings to datetime objects
    date_times = np.array([parse_datetime_string(dt_str) for dt_str in fileInstantiateTime])
    
    # first starts at 0
    starts = list(map(lambda x: x.seconds+x.microseconds/1e6,date_times-date_times[0]))

    timestamps = []

    for start,fs,framect in zip(starts,frameRates,frameCounts):
        timestamps.extend((np.arange(framect)/fs)+start)
    
    return timestamps,date_times