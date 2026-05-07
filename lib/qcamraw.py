"""
I/O and processing for QImaging .qcamraw files.

Mirrors logic from matlabPAC_process2P/GUIs/meanFluoROIvt.m:
  - parse ASCII header
  - read uint16 little-endian frames
  - compute spatial dF/F map (low-pass Butterworth) for ROI selection
  - extract mean ROI fluorescence trace (vectorised, matches
    lib.mat2py.getROIfluoFromTifs convention)
"""
import os
import re
import glob
import json
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy.signal import butter, filtfilt

from lib.mat2py import _interactive_environment_available


# -------------------------------------------------------------------- I/O ----

def read_qcamraw(path):
    """Read a .qcamraw file.

    Returns
    -------
    movie : (nFrames, height, width) uint16
    header : dict
        Parsed header fields, all values as strings.
    """
    header = {}
    with open(path, 'rb') as fid:
        while True:
            raw = fid.readline()
            if raw == b'':
                raise IOError(f'Unexpected EOF in header of {path}')
            line = raw.decode('ascii', errors='replace').rstrip('\r\n')
            if line == '':
                break
            key, _, val = line.partition(':')
            key = key.strip().replace('-', '_')
            val = val.strip()
            # strip a trailing bracketed-unit token like "[bytes]" or "[ns]"
            val = re.sub(r'\s*\[[^\]]*\]\s*$', '', val)
            header[key] = val

        header_size = int(header['Fixed_Header_Size'])
        roi = [int(x) for x in re.split(r'[,\s]+', header['ROI'].strip()) if x]
        img_w, img_h = roi[2], roi[3]
        frame_size = int(header['Frame_Size'])

        fid.seek(0, os.SEEK_END)
        n_bytes = fid.tell()
        n_frames = (n_bytes - header_size) // frame_size

        if (n_bytes - header_size) % frame_size != 0:
            raise IOError(
                f'{path}: data section is not a whole multiple of '
                f'frame size ({frame_size} bytes).'
            )

        fid.seek(header_size)
        raw = np.fromfile(fid, dtype='<u2',
                          count=img_w * img_h * n_frames)

    if raw.size != img_w * img_h * n_frames:
        raise IOError(f'Pixel count mismatch in {path}')

    # NumPy is row-major; reshape (F, H, W) directly.
    # NOTE: verify against a known frame on first use — if the image looks
    # transposed, swap to: raw.reshape(n_frames, img_w, img_h).transpose(0, 2, 1)
    movie = raw.reshape(n_frames, img_h, img_w)
    return movie, header


def get_qcamraw_start_time(path, header_field=None):
    """Acquisition start time from the qcamraw header.

    Tries `File_Init_Timestamp` (QCapture format: MM-DD-YYYY_HH:MM:SS),
    then ISO-8601 on a couple of fallback fields, then file mtime.
    """
    _, header = read_qcamraw(path)

    candidate_fields = ([header_field] if header_field
                        else ['File_Init_Timestamp',
                              'Header_Creation_Timestamp'])

    for field in candidate_fields:
        if not field or field not in header:
            continue
        val = header[field]
        # QCapture format
        try:
            dt = datetime.strptime(val, '%m-%d-%Y_%H:%M:%S')
            return dt.astimezone()  # interprets naive dt as local time, attaches local tz
        except ValueError:
            pass
        # ISO-8601 fallback in case other QCapture versions write it
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    warnings.warn(
        f'No parseable timestamp in {os.path.basename(path)} header; '
        f'falling back to file mtime. Header keys: {list(header.keys())}',
        RuntimeWarning,
    )
    mtime = os.path.getmtime(path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc)

def header_summary(header):
    """One-line comment string with key acquisition settings."""
    parts = []
    for k, label in [('Exposure', 'exposure'),
                     ('Spatial_Binning', 'binning'),
                     ('Normalized_Gain', 'gain'),
                     ('Image_Format', 'format')]:
        if k in header:
            parts.append(f'{label}={header[k]}')
    return ', '.join(parts)

# ------------------------------------------------------------- processing ----

def spatial_dff_map(movie, fr, baseline_window_s,
                    stim_len_s, framespan_frames,
                    filt_order=4, cutoff_hz=5.0):
    """Spatial dF/F map averaged over `framespan_frames` after stimulus.

    Mirrors meanFluoROIvt.m exactly: baseline-subtract → low-pass Butterworth
    along the frame axis → average frames in
    [baseline_end + stim_len, baseline_end + stim_len + framespan/fr].

    Raises ValueError if either window is empty (recording too short, or
    config doesn't match the actual timing).
    """
    movie = movie.astype(np.float64)
    n_frames = movie.shape[0]
    # MATLAB t = (1:N)/fr; first sample at 1/fr
    t = np.arange(1, n_frames + 1) / fr

    base_idx = (t >= baseline_window_s[0]) & (t <= baseline_window_s[1])
    if not base_idx.any():
        raise ValueError(
            f'Baseline window {baseline_window_s} contains no frames '
            f'(recording is {n_frames / fr:.2f} s).'
        )
    baseline = movie[base_idx].mean(axis=0)
    baseline_safe = np.where(baseline == 0, np.nan, baseline)
    dff = (movie - baseline) / baseline_safe

    norm_cutoff = cutoff_hz / (fr / 2.0)
    b, a = butter(filt_order, norm_cutoff, btype='low')
    dff_filt = filtfilt(b, a, np.nan_to_num(dff, nan=0.0), axis=0)

    post_start = baseline_window_s[1] + stim_len_s
    post_end = post_start + framespan_frames / fr
    post_idx = (t >= post_start) & (t <= post_end)
    if not post_idx.any():
        raise ValueError(
            f'Post-stimulus window [{post_start:.2f}, {post_end:.2f}] s '
            f'contains no frames (recording is {n_frames / fr:.2f} s).'
        )
    return dff_filt[post_idx].mean(axis=0)


# ------------------------------------------------------------------- ROI ----

def rect_to_image_mask(roi, height, width):
    """roi = (row1, row2, col1, col2), inclusive (MATLAB convention)."""
    r1, r2, c1, c2 = roi
    mask = np.zeros((height, width), dtype=bool)
    mask[r1:r2 + 1, c1:c2 + 1] = True
    return mask


def mean_fluo_in_roi_vectorised(movie, masks):
    """Mean ROI fluorescence per frame.

    Mirrors the matrix-multiply pattern in lib.mat2py.getROIfluoFromTifs so
    qcamraw and motion-corrected-tif pipelines compute traces identically.

    Parameters
    ----------
    movie : (nFrames, H, W) ndarray
    masks : (nROIs, H, W) bool/uint8 ndarray
        Single-ROI input should be passed as (1, H, W).

    Returns
    -------
    response : (nFrames, nROIs) float64
    """
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    nROIs, H, W = masks.shape
    mask_bool = (masks > 0).reshape(nROIs, -1).T.astype(np.float32)  # (H*W, nROIs)
    mask_sums = mask_bool.sum(axis=0)
    nF = movie.shape[0]
    img_flat = movie.reshape(nF, -1).astype(np.float32)
    return ((img_flat @ mask_bool) / mask_sums).astype(np.float64)


def _find_joblib_mask(exp_dir, treatment=''):
    """Return a joblib ROI mask array from exp_dir, or None if absent.

    Priority:
      1. *response_mask*.joblib whose filename contains the treatment key
         ('post' / 'pre' derived from treatment, or the treatment itself)
      2. Any *response_mask*.joblib (general, no treatment specificity)

    *contour* files are excluded to match the signalProcess.py convention.
    """
    try:
        import joblib
    except ImportError:
        warnings.warn(
            'joblib not installed — skipping joblib ROI mask lookup.',
            RuntimeWarning,
        )
        return None

    all_masks = glob.glob(os.path.join(exp_dir, '*response_mask*.joblib'))
    all_masks = [f for f in all_masks
                 if 'contour' not in os.path.basename(f).lower()]

    if not all_masks:
        return None

    if treatment:
        treat_lower = treatment.lower()
        if 'post' in treat_lower:
            treat_key = 'post'
        elif 'pre' in treat_lower:
            treat_key = 'pre'
        else:
            treat_key = treat_lower

        treat_masks = [f for f in all_masks
                       if treat_key in os.path.basename(f).lower()]
        if treat_masks:
            path = sorted(treat_masks)[0]
            print(f'  joblib ROI mask (treatment={treatment!r}): '
                  f'{os.path.basename(path)}')
            return joblib.load(path)

    path = sorted(all_masks)[0]
    print(f'  joblib ROI mask (general): {os.path.basename(path)}')
    return joblib.load(path)


def load_or_select_roi(qcamraw_path, movie, fr, cfg,
                       treatment='', save=True):
    """Resolve ROI for a qcamraw file via fallback ladder:

      1. *response_mask*.joblib in the experiment directory
         (treatment-specific first, then general)
      2. {basename}_qcamROI.json sidecar
      3. Interactive matplotlib RectangleSelector — on the spatial dF/F map
         if the recording timing supports it, otherwise on the first frame
         (matches the .tif branch of meanFluoROIvt.m)

    Returns (roi_tuple_or_None, mask, reference_image_or_None)
    For joblib sources roi_tuple is None (mask is a full 2-D array, not a
    rectangular bounding box). For JSON and interactive sources it is a tuple.
    """
    exp_dir = os.path.dirname(qcamraw_path)
    h, w = movie.shape[1:]

    joblib_mask = _find_joblib_mask(exp_dir, treatment=treatment)
    if joblib_mask is not None:
        mask = joblib_mask.astype(bool)
        if mask.shape != (h, w):
            warnings.warn(
                f'joblib mask shape {mask.shape} does not match frame '
                f'shape ({h}, {w}) — falling through to next source.',
                RuntimeWarning,
            )
        else:
            return None, mask, None

    sidecar = os.path.splitext(qcamraw_path)[0] + '_qcamROI.json'
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            roi = tuple(json.load(f)['roi'])
        return roi, rect_to_image_mask(roi, h, w), None

    # No joblib mask, no sidecar — interactive selection is the last resort.
    # In headless contexts (no DISPLAY, no TTY) raise an actionable error
    # rather than letting matplotlib fail opaquely.
    if not _interactive_environment_available():
        raise FileNotFoundError(
            f"\nCannot resolve ROI for {os.path.basename(qcamraw_path)}.\n"
            f"  No *response_mask*.joblib in {exp_dir or '.'}\n"
            f"  No sidecar at {sidecar}\n"
            f"  No DISPLAY for interactive RectangleSelector.\n\n"
            f"To proceed in a headless environment, create the sidecar:\n"
            f"  {sidecar}\n"
            f"with contents:\n"
            f'  {{"roi": [row1, row2, col1, col2], '
            f'"note": "[row1, row2, col1, col2], inclusive"}}\n'
            f"where row/col bounds index into the {h}x{w} qcamraw frame.\n"
        )

    # Try dF/F map; fall back to first frame if recording is too short or
    # baseline/stim config doesn't match.
    reference = None
    cmap = 'gray'
    try:
        reference = spatial_dff_map(
            movie, fr,
            baseline_window_s=cfg['baseline'],
            stim_len_s=cfg['stimlen'],
            framespan_frames=cfg['temporalAvgFrameWindow'],
            filt_order=cfg['filtOrder'],
            cutoff_hz=cfg['filtCutoffFreq'],
        )
        cmap = 'jet'
    except ValueError as e:
        warnings.warn(
            f'{os.path.basename(qcamraw_path)}: dF/F map unavailable '
            f'({e}); using first frame for ROI selection.',
            RuntimeWarning,
        )
        reference = movie[0]

    roi = _interactive_rect(reference, cmap=cmap,
                            title=os.path.basename(qcamraw_path))

    if save:
        with open(sidecar, 'w') as f:
            json.dump({'roi': list(roi),
                       'note': '[row1, row2, col1, col2], inclusive'}, f)
    return roi, rect_to_image_mask(roi, h, w), reference


def _interactive_rect(image, cmap='jet', title=''):
    """Matplotlib RectangleSelector → (row1, row2, col1, col2) inclusive."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    selected = {}

    def onselect(eclick, erelease):
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        selected['roi'] = (int(round(y1)), int(round(y2)),
                           int(round(x1)), int(round(x2)))

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    ax.set_title(f'Draw ROI for {title}, then close window')
    _ = RectangleSelector(ax, onselect, useblit=True,
                          button=[1], minspanx=2, minspany=2,
                          interactive=True)
    plt.show(block=True)
    if 'roi' not in selected:
        raise RuntimeError('No ROI selected.')
    return selected['roi']