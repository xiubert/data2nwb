"""
matchTifXSG.py
Match ScanImage .tif files to Ephus .xsg files by creation timestamp and
extract pulse metadata. Outputs pulseLegend2P.csv, which can be used as a
fallback by data2nwb when no *_Pulses.mat files are present.

Matching logic (default):
  - Each tif owns all xsg files created AFTER its start time and BEFORE
    the next tif's start time.
  - 1 matched xsg  → type = 'stim'
  - >1 matched xsg → type = 'map' (one row per xsg, tif repeated)

Matching logic (--one-per-tif):
  - Each tif is matched to only the single nearest xsg created at or after
    its start time.
  - Always type = 'stim' (one xsg per tif).

Usage:
    python extra/matchTifXSG.py /path/to/experiment/dir
    python extra/matchTifXSG.py /path/to/experiment/dir --tif-pattern "CC0002AAAA*.tif"
    python extra/matchTifXSG.py /path/to/experiment/dir --one-per-tif
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
    stimDelay and ISI are not in the xsg header — left as NaN for manual entry.
    """
    d = loadmat(xsg_path, squeeze_me=True, simplify_cells=True)
    stim = d['header']['stimulator']['stimulator']

    def _first_str(val):
        """Get first element of a pulseNameArray / pulseSetNameArray."""
        if isinstance(val, (list, )):
            return str(val[0]) if val else ''
        try:
            # numpy array of strings
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
# Matching
# ---------------------------------------------------------------------------

def matchTifsToXSGs(exp_dir: str, tif_pattern: str = '*.tif',
                    one_per_tif: bool = False) -> list[dict]:
    """
    For each tif in exp_dir, match xsg files by creation timestamp.

    Default: each tif owns all xsg files created between its start time and
    the next tif's start time (1 xsg → stim, >1 → map).

    one_per_tif=True: each tif is matched to only the nearest single xsg
    created at or after its start time (always type = 'stim').

    Returns list of row dicts with keys matching pulseLegend2P.csv columns:
        tif, type, pulseName, pulseSet, stimDelay, ISI, xsg
    stimDelay and ISI are set to '' (blank) as they are not in the xsg header.
    """
    exp_dir = Path(exp_dir)

    tif_paths = sorted(exp_dir.glob(tif_pattern))
    xsg_paths = sorted(exp_dir.glob('*.xsg'))

    if not tif_paths:
        raise FileNotFoundError(f"No tif files matching '{tif_pattern}' in {exp_dir}")
    if not xsg_paths:
        raise FileNotFoundError(f"No .xsg files found in {exp_dir}")

    print(f"Found {len(tif_paths)} tif(s) and {len(xsg_paths)} xsg(s)")

    # get tif timestamps
    print("Reading tif timestamps...")
    tif_times = []
    for p in tif_paths:
        try:
            t = lib.tifExtract.getSItifTime(str(p))
            tif_times.append((p, t))
            print(f"  {p.name}  {t.strftime('%d-%b-%Y %H:%M:%S')}")
        except Exception as e:
            print(f"  WARNING: could not read time from {p.name}: {e}")

    # get xsg timestamps
    print("Reading xsg timestamps...")
    xsg_times = []
    for p in xsg_paths:
        try:
            t = getXSGtime(str(p))
            xsg_times.append((p, t))
            print(f"  {p.name}  {t.strftime('%d-%b-%Y %H:%M:%S')}")
        except Exception as e:
            print(f"  WARNING: could not read time from {p.name}: {e}")

    # match: each tif owns xsgs between its time and the next tif's time,
    # or (one_per_tif) only the nearest single xsg at/after its time
    rows = []
    for i, (tif_path, tif_t) in enumerate(tif_times):
        if one_per_tif:
            candidates = [(xp, xt) for xp, xt in xsg_times if xt >= tif_t]
            if not candidates:
                print(f"  WARNING: no xsg matched for {tif_path.name}")
                continue
            matched_xsgs = [min(candidates, key=lambda x: x[1])]
            tif_type = 'stim'
        else:
            next_tif_t = tif_times[i + 1][1] if i + 1 < len(tif_times) else datetime.max
            matched_xsgs = [
                (xp, xt) for xp, xt in xsg_times
                if tif_t <= xt < next_tif_t
            ]
            if not matched_xsgs:
                print(f"  WARNING: no xsg matched for {tif_path.name}")
                continue
            tif_type = 'map' if len(matched_xsgs) > 1 else 'stim'

        print(f"  {tif_path.name} → {len(matched_xsgs)} xsg(s) ({tif_type})")

        for xsg_path, _ in matched_xsgs:
            try:
                pulse = getXSGpulseInfo(str(xsg_path))
            except Exception as e:
                print(f"    WARNING: could not read pulse info from {xsg_path.name}: {e}")
                pulse = {'pulseName': '', 'pulseSet': ''}

            rows.append({
                'tif':       tif_path.name,
                'type':      tif_type,
                'pulseName': pulse['pulseName'],
                'pulseSet':  pulse['pulseSet'],
                'stimDelay': '',   # not available in xsg header — fill manually
                'ISI':       '',   # not available in xsg header — fill manually
                'xsg':       xsg_path.name,
            })

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exp_dir', help='Path to experiment directory')
    parser.add_argument('--tif-pattern', default='*.tif',
                        help='Glob pattern for tif files (default: *.tif)')
    parser.add_argument('--one-per-tif', action='store_true',
                        help='Match only the nearest single xsg per tif (always stim); '
                             'default matches all xsgs between consecutive tifs')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: exp_dir/pulseLegend2P.csv)')
    args = parser.parse_args()

    rows = matchTifsToXSGs(args.exp_dir, args.tif_pattern, one_per_tif=args.one_per_tif)

    if not rows:
        print("No rows produced — check warnings above.")
        sys.exit(1)

    out_path = args.output or os.path.join(args.exp_dir, 'pulseLegend2P.csv')
    fieldnames = ['tif', 'type', 'pulseName', 'pulseSet', 'stimDelay', 'ISI', 'xsg']

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {out_path}")
    print("Note: stimDelay and ISI columns are blank — fill manually if needed.")


if __name__ == '__main__':
    main()
