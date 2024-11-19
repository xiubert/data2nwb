"""
Helpers for grabbing image data and metadata from ScanImage .tif files
"""
from datetime import datetime
from ScanImageTiffReader import ScanImageTiffReader
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import (
            extract_extra_metadata,
            parse_metadata,
        )


def getSItifTime(tifFile: str) -> datetime:
    """
    Extracts tif timestamp from tif file.

    Args:
        tifFile (str): path of .tif file

    Returns:
        (datetime) timestamp of tif file instantiation
    """

    image_metadata = extract_extra_metadata(file_path=tifFile)
    fileTimeInstantiate = image_metadata['epoch']
    return parse_datetime_string(fileTimeInstantiate)


def getSItifData(tifFile: str, getMetadata: bool = True):
    """
    Grabs .tif metadata as including frame rate, file instantiation time (creation time), 
    and frame number from ScanImage .tif file list.

    Args:
        tifFile (str): path of .tif file
        getMetadata (bool): whether to include tif metadata in output

    Returns:
        imgData: 3d numpy array (frame) x (X) x (Y)
        fileTimeInstantiate (list): timestamp of tif file instantiation (as list for multiple tifs)
        nFrames (list): number of frames in each tif file
        frameRate (list): framerate at which tif file was collected
    """
    imgData = ScanImageTiffReader(tifFile).data()

    if getMetadata:
        image_metadata = extract_extra_metadata(file_path=tifFile)
        fileTimeInstantiate = parse_datetime_string(image_metadata['epoch'])
        extraMeta = parse_metadata(image_metadata)
        frameRate = extraMeta['sampling_frequency']
        nFrames = extraMeta['frames_per_slice']
    
        return imgData,fileTimeInstantiate,nFrames,frameRate
    
    return imgData
    

def parse_datetime_string(date_str: str) -> datetime:
    """
    Parse date string from ScanImage .tif metadata ('epoch') and convert it to a datetime object.

    Args:
        date_str: date string contained in 'epoch' variable of .tif metadata. Eg. 'epoch': '[2021,3,11,20,3,13.185]'
    
    Returns:
        dt: datetime object to the microsecond resolution
    """

    # Remove the brackets and split the string by commas
    date_str = date_str.strip('[]')
    date_list = [float(x) for x in date_str.split(',')]
    
    # Convert the list into a datetime object
    dt = parse_datetime_list(date_list)

    return dt


def parse_datetime_list(date_list: list[float]) -> datetime:
    """
    Takes list from parsed ScanImage tif epoch time string (in parse_datetime_string) and parses list to a datetime object.

    Args:
        date_list: date string from ScanImage .tif 'epoch' variable as list of floats. Eg. [2021, 3, 11, 20, 3, 13.185]
    
    Returns:
        dt: datetime object to the microsecond resolution
    """
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


def secMicroSec2sec(datetime: datetime):
    """
    Returns seconds from datetime object split into seconds and microseconds.

    Args:
        datetime: datetime object with microsecond resolution
    
    Returns:
        dt: datetime object with fractional second resolution
    """
    return datetime.seconds+datetime.microseconds/1e6