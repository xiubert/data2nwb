"""
Helpers for grabbing image data and metadata from ScanImage .tif files
"""
from datetime import datetime
from ScanImageTiffReader import ScanImageTiffReader
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import (
            extract_extra_metadata,
            parse_metadata,
        )

# Track which legacy ScanImage versions we've already announced so the notice
# fires once per (version, minorRev) per process instead of for every tif.
_LEGACY_NOTICE_SHOWN: set[tuple[str, str]] = set()


def _is_legacy_metadata(raw: dict) -> bool:
    """Legacy SI (~3.x) uses 'state.*' keys and has no 'epoch'; modern SI uses 'SI.*'."""
    return 'epoch' not in raw and any(k.startswith('state.') for k in raw)


def _strip_si_quotes(value: str) -> str:
    """Legacy SI header values are wrapped in literal single-quotes, e.g. "'5.008'"."""
    if isinstance(value, str) and len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def _notice_legacy_version(raw: dict, tifFile: str) -> None:
    version = _strip_si_quotes(raw.get('state.software.version', '?'))
    minor = _strip_si_quotes(raw.get('state.software.minorRev', '?'))
    key = (version, minor)
    if key in _LEGACY_NOTICE_SHOWN:
        return
    _LEGACY_NOTICE_SHOWN.add(key)
    print(
        f"[tifExtract] NOTICE: legacy ScanImage TIFF detected "
        f"(version {version}.{minor}, 'state.*' header schema, no 'epoch' field). "
        f"Using legacy parser. First file: {tifFile}"
    )


def parse_legacy_datetime_string(date_str: str) -> datetime:
    """
    Parse legacy ScanImage trigger-time string, e.g. "'11/18/2024 16:42:29.375'",
    into a datetime with microsecond resolution.
    """
    s = _strip_si_quotes(date_str.strip())
    return datetime.strptime(s, "%m/%d/%Y %H:%M:%S.%f")


def _parse_legacy_metadata(raw: dict) -> dict:
    """
    Normalize legacy 'state.*' header values into the same keys that
    roiextractors' parse_metadata produces for modern files.
    """
    return {
        'sampling_frequency': float(_strip_si_quotes(raw['state.acq.frameRate'])),
        'frames_per_slice': int(_strip_si_quotes(raw['state.acq.numberOfFrames'])),
    }


def _legacy_session_time(raw: dict) -> datetime:
    """
    Pick the legacy timestamp closest in meaning to modern 'epoch'.
    Prefer triggerTimeFirstString (first trigger of acquisition); fall back to
    triggerTimeString, then softTriggerTimeString.
    """
    for key in ('state.internal.triggerTimeFirstString',
                'state.internal.triggerTimeString',
                'state.internal.softTriggerTimeString'):
        if key in raw and _strip_si_quotes(raw[key]).strip():
            return parse_legacy_datetime_string(raw[key])
    raise KeyError(
        "legacy ScanImage tif has no usable trigger-time field "
        "(triggerTimeFirstString / triggerTimeString / softTriggerTimeString)"
    )


def getSItifMetadata(tifFile: str, parse: bool = False) -> dict:
    """
    Extract metadata from a ScanImage .tif file.

    Args:
        tifFile (str): path to the .tif file
        parse (bool): if True, return a normalized dict with 'sampling_frequency'
                      and 'frames_per_slice'; if False, return the raw header dict
                      (modern: 'epoch', 'SI.*'; legacy: 'state.*').

    Returns:
        dict: metadata key-value pairs from the ScanImage header
    """
    raw = extract_extra_metadata(file_path=tifFile)
    if not parse:
        return raw
    if _is_legacy_metadata(raw):
        _notice_legacy_version(raw, tifFile)
        return _parse_legacy_metadata(raw)
    return parse_metadata(raw)


def getSItifTime(tifFile: str) -> datetime:
    """
    Extracts tif timestamp from tif file.

    Args:
        tifFile (str): path of .tif file

    Returns:
        (datetime) timestamp of tif file instantiation
    """

    image_metadata = extract_extra_metadata(file_path=tifFile)
    if _is_legacy_metadata(image_metadata):
        _notice_legacy_version(image_metadata, tifFile)
        return _legacy_session_time(image_metadata)
    return parse_datetime_string(image_metadata['epoch'])


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
        if _is_legacy_metadata(image_metadata):
            _notice_legacy_version(image_metadata, tifFile)
            fileTimeInstantiate = _legacy_session_time(image_metadata)
            extraMeta = _parse_legacy_metadata(image_metadata)
        else:
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
