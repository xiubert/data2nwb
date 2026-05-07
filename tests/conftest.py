"""
Shared test utilities for data2nwb integration tests.

Tests require a populated experiment directory. Set the env var
DATA2NWB_TEST_DATA to point to the parent path containing per-subject
folders (CC0001/, CC0002/, ...). Defaults to the dev location used while
authoring these tests.

Tests pytest-compatible (`pytest tests/`) and also runnable as scripts
(`python tests/test_qcam.py`).
"""
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TEST_DATA = (
    "/media/DATA/backups/sutter2P_backup/D_drive/offsetCTRL/nwb_test/"
)


def get_test_data_path() -> Path:
    """Resolve test data path from env var or default. Returns None if absent."""
    p = Path(os.environ.get("DATA2NWB_TEST_DATA", DEFAULT_TEST_DATA))
    return p if p.is_dir() else None


def require_test_data():
    """Skip-or-fail helper: returns the test data path or raises a skip-style
    error suitable for both pytest and standalone runs.
    """
    p = get_test_data_path()
    if p is None:
        msg = (
            f"test data not found. Set DATA2NWB_TEST_DATA or place test "
            f"data at {DEFAULT_TEST_DATA}"
        )
        try:
            import pytest
            pytest.skip(msg, allow_module_level=False)
        except ImportError:
            raise RuntimeError(msg)
    return p


def require_subject_dir(subject_id: str) -> Path:
    """Resolve {test_data}/{subject_id}, skipping if absent."""
    base = require_test_data()
    sub = base / subject_id
    if not sub.is_dir():
        msg = f"subject directory missing: {sub}"
        try:
            import pytest
            pytest.skip(msg, allow_module_level=False)
        except ImportError:
            raise RuntimeError(msg)
    return sub


def load_config(name: str) -> dict:
    """Load a YAML config from configs/."""
    with open(REPO_ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def load_subjects_csv() -> "pandas.DataFrame":
    import pandas as pd
    return pd.read_csv(REPO_ROOT / "example_data" / "animalList.csv").set_index(
        "subject_id"
    )


def load_experiments_csv() -> "pandas.DataFrame":
    import ast
    import pandas as pd
    df = pd.read_csv(
        REPO_ROOT / "example_data" / "experimentMetadata.csv"
    ).set_index("subject_id")
    df["keywords"] = df["keywords"].apply(ast.literal_eval)
    return df


def run_qcam_pipeline(subject_id: str) -> Path:
    """Run the qcam pipeline against the test data and return the NWB path."""
    import lib.nwbScanImage
    import lib.nwbQcam

    base = require_test_data()
    require_subject_dir(subject_id)

    cfg = load_config("params_qcam.yaml")
    subjects = load_subjects_csv()
    experiments = load_experiments_csv()

    out_path = base / subject_id / f"{subject_id}_qcam_DANDI.nwb"
    subject = lib.nwbScanImage.setSubject(
        subject_id=subject_id,
        age=f"P{subjects.loc[subject_id]['age']}D",
        species="Mus musculus",
        sex=subjects.loc[subject_id]["sex"],
        genotype=subjects.loc[subject_id]["genotype"],
        description=subjects.loc[subject_id]["description"],
    )
    d = experiments.loc[subject_id]
    lib.nwbQcam.genNWBfromQcamraw_pc(
        experimentID=subject_id,
        dataPath=str(base),
        NWBoutputPath=str(out_path),
        subject=subject,
        session_description=d["session_description"],
        experiment_description=d["experiment_description"],
        keywords=d["keywords"],
        **cfg["nwb_file"],
        **cfg["imaging"],
        **cfg["qcam"],
    )
    return out_path


def run_2p_pipeline(subject_id: str) -> Path:
    """Run the 2P pipeline against the test data and return the NWB path."""
    import lib.nwbScanImage

    base = require_test_data()
    require_subject_dir(subject_id)

    cfg = load_config("params_PC.yaml")
    subjects = load_subjects_csv()
    experiments = load_experiments_csv()

    out_path = base / subject_id / f"{subject_id}_2P_DANDI.nwb"
    subject = lib.nwbScanImage.setSubject(
        subject_id=subject_id,
        age=f"P{subjects.loc[subject_id]['age']}D",
        species="Mus musculus",
        sex=subjects.loc[subject_id]["sex"],
        genotype=subjects.loc[subject_id]["genotype"],
        description=subjects.loc[subject_id]["description"],
    )
    d = experiments.loc[subject_id]
    lib.nwbScanImage.genNWBfromScanImage_pc(
        experimentID=subject_id,
        dataPath=str(base),
        NWBoutputPath=str(out_path),
        subject=subject,
        session_description=d["session_description"],
        experiment_description=d["experiment_description"],
        keywords=d["keywords"],
        **cfg["nwb_file"],
        **cfg["imaging"],
        **cfg.get("dataProcessing", {}),
    )
    return out_path


def run_standalone(test_funcs):
    """Run a list of test functions sequentially, printing pass/fail.

    Used in `if __name__ == '__main__'` blocks so test files run without
    pytest installed. Returns nonzero exit code on first failure.
    """
    failed = 0
    for fn in test_funcs:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(test_funcs) - failed}/{len(test_funcs)} passed")
    sys.exit(0 if failed == 0 else 1)
