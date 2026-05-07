"""
Integration tests for the qcam (.qcamraw → NWB) pipeline.

Runs the full pipeline once for CC0001 then checks structural correctness
of the resulting NWB (stim table, OnePhotonSeries, ImagingPlane, ROI
segmentation, and nwbinspector compliance for known-clean checks).

Run with:
    pytest tests/test_qcam.py            # if pytest installed
    python tests/test_qcam.py            # standalone
"""
import sys
from datetime import datetime
from pathlib import Path

# Make `import conftest` work whether invoked via pytest or as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pynwb import NWBHDF5IO

from conftest import (
    require_subject_dir,
    run_qcam_pipeline,
    run_standalone,
)

SUBJECT_ID = "CC0001"

EXPECTED_STIM_COLUMNS = [
    "file",
    "OnePhotonSeries",
    "fileTimeInstantiate",
    "starting_time",
    "type",
    "nFrames",
    "frameRate",
    "treatment",
    "pulseNames",
    "pulseSets",
    "ISI",
    "stimDelay",
    "xsg",
]


# --------------------------------------------------------------- helpers ----

_NWB_CACHE = {}


def _nwb_path() -> Path:
    """Run pipeline once per session, cache path."""
    if "qcam" not in _NWB_CACHE:
        _NWB_CACHE["qcam"] = run_qcam_pipeline(SUBJECT_ID)
    return _NWB_CACHE["qcam"]


# ----------------------------------------------------------------- tests ----

def test_pipeline_produces_nwb():
    p = _nwb_path()
    assert p.exists(), f"Pipeline did not produce {p}"
    assert p.stat().st_size > 0, f"Output {p} is empty"


def test_session_start_time_set():
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        assert nwb.session_start_time is not None


def test_one_photon_series_count_matches_qcamraw_files():
    sub_dir = require_subject_dir(SUBJECT_ID)
    qcam_files = sorted(sub_dir.glob(f"{SUBJECT_ID}*.qcamraw"))
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        ops = [n for n in nwb.acquisition if n.startswith("OnePhotonSeries_")]
        assert len(ops) == len(qcam_files), (
            f"OnePhotonSeries count {len(ops)} != qcamraw count {len(qcam_files)}"
        )


def test_imaging_plane_has_required_fields():
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        planes = list(nwb.imaging_planes.values())
        assert len(planes) >= 1, "no ImagingPlane found"
        plane = planes[0]
        assert plane.indicator
        assert plane.excitation_lambda > 0
        assert plane.imaging_rate > 0


def test_stim_table_present_and_columns():
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        stim = nwb.get_stimulus("stim param table")
        assert stim is not None, "stim param table missing"
        cols = list(stim.colnames)
        assert cols == EXPECTED_STIM_COLUMNS, (
            f"stim table columns:\n  got:      {cols}\n  expected: {EXPECTED_STIM_COLUMNS}"
        )


def test_stim_table_treatment_matches_pulse_legend():
    """When pulseLegendQcam.csv exists, treatment column should match its values."""
    sub_dir = require_subject_dir(SUBJECT_ID)
    legend_csv = sub_dir / "pulseLegendQcam.csv"
    if not legend_csv.exists():
        # acceptable: no legend → treatments come from file list, may be empty
        return
    import csv
    with open(legend_csv) as f:
        reader = csv.DictReader(f)
        legend_treatments = [r.get("treatment", r.get("condition", "")) for r in reader]
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        stim = nwb.get_stimulus("stim param table")
        nwb_treatments = list(stim["treatment"][:])
    assert nwb_treatments == legend_treatments, (
        f"treatment mismatch:\n  nwb:    {nwb_treatments}\n  legend: {legend_treatments}"
    )


def test_filetime_instantiate_parseable():
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        stim = nwb.get_stimulus("stim param table")
        for ts in stim["fileTimeInstantiate"][:]:
            datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")  # raises on bad format


def test_starting_time_first_zero_or_positive():
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        stim = nwb.get_stimulus("stim param table")
        starts = list(stim["starting_time"][:])
        # the earliest acquisition should anchor session_start_time, so the
        # smallest starting_time is 0.0; all others non-negative
        assert min(starts) == 0.0, f"min starting_time should be 0.0, got {min(starts)}"
        assert all(s >= 0 for s in starts)


def test_nframes_and_framerate_positive():
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        stim = nwb.get_stimulus("stim param table")
        assert all(int(n) > 0 for n in stim["nFrames"][:])
        assert all(float(r) > 0 for r in stim["frameRate"][:])


def test_roi_segmentation_present():
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        ophys = nwb.processing.get("ophys")
        assert ophys is not None
        img_seg = ophys.data_interfaces.get("ImageSegmentation")
        assert img_seg is not None
        assert len(img_seg.plane_segmentations) >= 1


def test_fluorescence_traces_match_one_photon_count():
    with NWBHDF5IO(str(_nwb_path()), "r") as io:
        nwb = io.read()
        ophys = nwb.processing["ophys"]
        fl = ophys.data_interfaces["Fluorescence"]
        ops = [n for n in nwb.acquisition if n.startswith("OnePhotonSeries_")]
        rrs = [n for n in fl.roi_response_series]
        assert len(rrs) == len(ops), (
            f"RoiResponseSeries count {len(rrs)} != OnePhotonSeries count {len(ops)}"
        )


def test_nwbinspector_no_critical_errors():
    """Run nwbinspector under DANDI config and assert no CRITICAL-or-worse issues.

    The check_data_orientation warning is a known false positive for our
    NumPy-row-major (frames, H, W) layout, so it is excluded explicitly.
    """
    try:
        from nwbinspector import inspect_nwbfile, Importance, load_config
    except ImportError:
        try:
            import pytest
            pytest.skip("nwbinspector not available")
        except ImportError:
            print("    (skipped: nwbinspector not installed)")
            return

    messages = list(inspect_nwbfile(
        nwbfile_path=str(_nwb_path()),
        config=load_config(filepath_or_keyword="dandi"),
    ))
    ignored_checks = {"check_data_orientation"}
    blocking = [
        m for m in messages
        if m.importance.value >= Importance.CRITICAL.value
        and m.check_function_name not in ignored_checks
    ]
    assert not blocking, (
        "nwbinspector reports critical issues:\n"
        + "\n".join(f"  - [{m.check_function_name}] {m.message}" for m in blocking)
    )


# -------------------------------------------------------------- standalone ----

if __name__ == "__main__":
    run_standalone([
        test_pipeline_produces_nwb,
        test_session_start_time_set,
        test_one_photon_series_count_matches_qcamraw_files,
        test_imaging_plane_has_required_fields,
        test_stim_table_present_and_columns,
        test_stim_table_treatment_matches_pulse_legend,
        test_filetime_instantiate_parseable,
        test_starting_time_first_zero_or_positive,
        test_nframes_and_framerate_positive,
        test_roi_segmentation_present,
        test_fluorescence_traces_match_one_photon_count,
        test_nwbinspector_no_critical_errors,
    ])
