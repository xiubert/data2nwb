"""
matchXSG.py
Match ScanImage .tif or QImaging .qcamraw files to Ephus .xsg files by
creation timestamp and extract pulse metadata.

Output:
  pulseLegend2P.csv   when matching .tif files
  pulseLegendQcam.csv when matching .qcamraw files

Matching logic (default):
  - Each file owns all xsg files created AFTER its start time and BEFORE
    the next file's start time.
  - 1 matched xsg  → type = 'stim'
  - >1 matched xsg → type = 'map' (one row per xsg, file repeated)

Matching logic (--one-per-file):
  - Each file is matched to only the single nearest xsg created at or after
    its start time.
  - Always type = 'stim' (one xsg per file).

Usage:
    python extra/matchXSG.py /path/to/experiment/dir
    python extra/matchXSG.py /path/to/experiment/dir --pattern "CC0001*.qcamraw"
    python extra/matchXSG.py /path/to/experiment/dir --one-per-file
"""

import sys
import os
import csv
import argparse
from datetime import datetime
from pathlib import Path
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib.tifExtract
import lib.qcamraw


# ---------------------------------------------------------------------------
# XSG helpers
# ---------------------------------------------------------------------------

def _squeeze(val):
    """Recursively unwrap 1-element numpy arrays to scalars."""
    try:
        while hasattr(val, '__len__') and len(val) == 1:
            val = val[0]
    except TypeError:
        pass
    return val


def getXSGtime(xsg_path: str) -> datetime:
    """
    Parse creation timestamp from an Ephus .xsg file.
    Timestamp is at header.xsg.xsg.xsgFileCreationTimestamp,
    formatted as e.g. '23-Mar-2022 12:51:39'.
    """
    d = loadmat(xsg_path, squeeze_me=True, simplify_cells=True)
    ts_str = d['header']['xsg']['xsg']['xsgFileCreationTimestamp']
    return datetime.strptime(str(ts_str).strip(), '%d-%b-%Y %H:%M:%S')


def getXSGpulseInfo(xsg_path: str) -> dict:
    """
    Extract pulse metadata from an Ephus .xsg header.

    Returns dict with keys: pulseName, pulseSet (empty string if unavailable).
    stimDelay and ISI are not in the xsg header — left blank for manual entry.
    """
    d = loadmat(xsg_path, squeeze_me=True, simplify_cells=True)
    stim = d['header']['stimulator']['stimulator']

    def _first_str(val):
        if isinstance(val, list):
            return str(val[0]) if val else ''
        try:
            return str(val.flat[0])
        except Exception:
            return str(val)

    pulse_name = ''
    pulse_set  = ''

    if 'pulseNameArray' in stim:
        pulse_name = _first_str(stim['pulseNameArray'])
    if 'pulseSetNameArray' in stim:
        pulse_set = _first_str(stim['pulseSetNameArray'])

    return {'pulseName': pulse_name, 'pulseSet': pulse_set}


# ---------------------------------------------------------------------------
# Timestamp dispatch
# ---------------------------------------------------------------------------

def _get_file_time(path: Path) -> datetime:
    """Return a naive datetime for path, dispatching on file extension."""
    if path.suffix.lower() == '.qcamraw':
        dt = lib.qcamraw.get_qcamraw_start_time(str(path))
        return dt.replace(tzinfo=None)
    return lib.tifExtract.getSItifTime(str(path))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def matchFilesToXSGs(exp_dir: str, pattern: str = '*.tif',
                     one_per_file: bool = False) -> list[dict]:
    """
    For each file matched by pattern in exp_dir, match xsg files by timestamp.

    Default: each file owns all xsg files created between its start time and
    the next file's start time (1 xsg → stim, >1 → map).

    one_per_file=True: each file is matched to only the nearest single xsg
    created at or after its start time (always type = 'stim').

    Returns list of row dicts with keys matching pulseLegend{2P,Qcam}.csv columns:
        file, type, pulseName, pulseSet, stimDelay, ISI, xsg, treatment
    stimDelay and ISI are set to '' (blank) as they are not in the xsg header.
    """
    exp_dir = Path(exp_dir)

    file_paths = sorted(exp_dir.glob(pattern))
    xsg_paths  = sorted(exp_dir.glob('*.xsg'))

    if not file_paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {exp_dir}")
    if not xsg_paths:
        raise FileNotFoundError(f"No .xsg files found in {exp_dir}")

    print(f"Found {len(file_paths)} file(s) and {len(xsg_paths)} xsg(s)")

    print("Reading file timestamps...")
    file_times = []
    for p in file_paths:
        try:
            t = _get_file_time(p)
            file_times.append((p, t))
            print(f"  {p.name}  {t.strftime('%d-%b-%Y %H:%M:%S')}")
        except Exception as e:
            print(f"  WARNING: could not read time from {p.name}: {e}")

    print("Reading xsg timestamps...")
    xsg_times = []
    for p in xsg_paths:
        try:
            t = getXSGtime(str(p))
            xsg_times.append((p, t))
            print(f"  {p.name}  {t.strftime('%d-%b-%Y %H:%M:%S')}")
        except Exception as e:
            print(f"  WARNING: could not read time from {p.name}: {e}")

    rows = []
    for i, (file_path, file_t) in enumerate(file_times):
        if one_per_file:
            candidates = [(xp, xt) for xp, xt in xsg_times if xt >= file_t]
            if not candidates:
                print(f"  WARNING: no xsg matched for {file_path.name}")
                continue
            matched_xsgs = [min(candidates, key=lambda x: x[1])]
            file_type = 'stim'
        else:
            next_file_t = file_times[i + 1][1] if i + 1 < len(file_times) else datetime.max
            matched_xsgs = [
                (xp, xt) for xp, xt in xsg_times
                if file_t <= xt < next_file_t
            ]
            if not matched_xsgs:
                print(f"  WARNING: no xsg matched for {file_path.name}")
                continue
            file_type = 'map' if len(matched_xsgs) > 1 else 'stim'

        print(f"  {file_path.name} → {len(matched_xsgs)} xsg(s) ({file_type})")

        for xsg_path, _ in matched_xsgs:
            try:
                pulse = getXSGpulseInfo(str(xsg_path))
            except Exception as e:
                print(f"    WARNING: could not read pulse info from {xsg_path.name}: {e}")
                pulse = {'pulseName': '', 'pulseSet': ''}

            rows.append({
                'file':      file_path.name,
                'type':      file_type,
                'pulseName': pulse['pulseName'],
                'pulseSet':  pulse['pulseSet'],
                'stimDelay': '',
                'ISI':       '',
                'xsg':       xsg_path.name,
                'treatment': '',
            })

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output(exp_dir: str, pattern: str) -> str:
    ext = Path(pattern).suffix.lower()
    name = 'pulseLegendQcam.csv' if ext == '.qcamraw' else 'pulseLegend2P.csv'
    return os.path.join(exp_dir, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exp_dir', help='Path to experiment directory')
    parser.add_argument('--pattern', default='*.tif',
                        help='Glob pattern for imaging files (default: *.tif); '
                             'use "*.qcamraw" for widefield data')
    parser.add_argument('--one-per-file', action='store_true',
                        help='Match only the nearest single xsg per file (always stim); '
                             'default matches all xsgs between consecutive files')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: exp_dir/pulseLegend2P.csv for .tif, '
                             'exp_dir/pulseLegendQcam.csv for .qcamraw)')
    args = parser.parse_args()

    rows = matchFilesToXSGs(args.exp_dir, args.pattern, one_per_file=args.one_per_file)

    if not rows:
        print("No rows produced — check warnings above.")
        sys.exit(1)

    out_path = args.output or _default_output(args.exp_dir, args.pattern)
    fieldnames = ['file', 'type', 'pulseName', 'pulseSet', 'stimDelay', 'ISI', 'xsg', 'treatment']

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {out_path}")
    print("Note: stimDelay and ISI columns are blank — fill manually if needed.")


if __name__ == '__main__':
    main()
