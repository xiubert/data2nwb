"""
Minimal smoke test for the qcamraw → NWB writer.

Reads one .qcamraw file, writes a minimal NWB with OnePhotonSeries +
PlaneSegmentation + RoiResponseSeries, runs nwbinspector against it.

Usage:
    python extra/qcam_smoke_test.py /path/to/file.qcamraw --fr 20
"""
import argparse
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject
from pynwb.ophys import (Fluorescence, ImageSegmentation,
                          OnePhotonSeries, OpticalChannel,
                          RoiResponseSeries)
from hdmf.backends.hdf5.h5_utils import H5DataIO

from lib.qcamraw import (read_qcamraw, rect_to_image_mask,
                          mean_fluo_in_roi_vectorised)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('qcamraw')
    ap.add_argument('--fr', type=float, default=20.0)
    ap.add_argument('--inspect', action='store_true',
                    help='Run nwbinspector after writing')
    args = ap.parse_args()

    movie, header = read_qcamraw(args.qcamraw)
    print(f'movie shape: {movie.shape}, dtype: {movie.dtype}')
    print(f'header keys: {list(header.keys())}')

    h, w = movie.shape[1:]
    # central 50×50 ROI
    roi = (h // 2 - 25, h // 2 + 24, w // 2 - 25, w // 2 + 24)
    mask = rect_to_image_mask(roi, h, w)
    trace = mean_fluo_in_roi_vectorised(movie, mask)

    nwbfile = NWBFile(
        session_description='qcamraw smoke test',
        identifier=str(uuid4()),
        session_start_time=datetime.now(tz=timezone.utc),
    )
    nwbfile.subject = Subject(subject_id='SMOKE', age='P1D',
                              species='Mus musculus', sex='U')

    device = nwbfile.create_device(name='Camera', description='QImaging')
    oc = OpticalChannel(name='OpticalChannel', description='filter',
                        emission_lambda=525.0)
    plane = nwbfile.create_imaging_plane(
        name='ImagingPlane', optical_channel=oc,
        imaging_rate=args.fr, description='widefield',
        device=device, excitation_lambda=470.0,
        indicator='GCaMP6f', location='ACtx',
    )

    op = OnePhotonSeries(
        name='OnePhotonSeries',
        data=H5DataIO(data=movie, compression=True),
        imaging_plane=plane, rate=args.fr, unit='n.a.',
    )
    nwbfile.add_acquisition(op)

    img_seg = ImageSegmentation()
    ps = img_seg.create_plane_segmentation(
        name='PlaneSegmentation',
        description='central rectangle',
        imaging_plane=plane, reference_images=[op],
    )
    ps.add_roi(image_mask=mask.astype(np.uint8))

    mod = nwbfile.create_processing_module(name='ophys', description='ophys')
    mod.add(img_seg)

    fl = Fluorescence()
    mod.add(fl)                                # Fluorescence attached to file

    rt = ps.create_roi_table_region(region=[0], description='central')
    fl.create_roi_response_series(             # use the factory method
        name='RoiResponseSeries',
        data=trace, rois=rt, unit='n.a.', rate=args.fr,
    )

    with tempfile.NamedTemporaryFile(suffix='.nwb', delete=False) as f:
        out = f.name
    with NWBHDF5IO(out, 'w') as io:
        io.write(nwbfile)
    print(f'wrote {out}')

    if args.inspect:
        sys.exit(subprocess.call(['nwbinspector', out, '--config', 'dandi']))


if __name__ == '__main__':
    main()