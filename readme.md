# data2nwb — ScanImage 2P & QImaging widefield ophys to NWB converter

Converts calcium imaging data to [NWB](https://www.nwb.org/) format for upload to [DANDI](https://dandiarchive.org/).

Two pipelines are provided, sharing subject/experiment metadata:

- **`scanimage2nwb.py`** — ScanImage two-photon `.tif` data (with motion correction, optional pupillometry).
- **`qcam2nwb.py`** — QImaging widefield epifluorescence `.qcamraw` data, written as `OnePhotonSeries`.

---

## Setup

> **Python version:** best compatibility with **Python 3.12**. On Windows, install it from the Microsoft Store (search "Python 3.12") — this exposes the `python3.12` launcher used below.

1. Clone the repository and change into it:

   ```bash
   git clone https://github.com/xiubert/data2nwb.git
   cd data2nwb
   ```

2. Create and activate a virtual environment (use `python3.12` explicitly to ensure 3.12 is used even if other versions are installed):

   ```bash
   python3.12 -m venv env
   source env/bin/activate          # unix
   source env/Scripts/activate      # Windows (Git Bash)
   env\Scripts\activate.bat         # Windows (cmd.exe)
   ```

3. Install dependencies (once the env is activated, `pip` is the env's pip — use it normally):

   ```bash
   pip install -r requirements.txt
   ```

---

## Shared metadata

Both pipelines read subject and experiment metadata from CSVs stored in `data/` by default. See `data/animalList.csv` and `data/experimentMetadata.csv` for examples. `subject_id` must match across both CSVs and the corresponding data folder name.

**`animalList.csv`** — one row per subject:

| Column | Required | Description |
| --- | --- | --- |
| `subject_id` | yes | Animal identifier; must match `experimentID` and data folder name |
| `age` | yes | Age at experiment in days (integer); written as `P{age}D` (ISO 8601) |
| `sex` | yes | `M` or `F` |
| `genotype` | yes | e.g. `C57BL6/J`, `ZnT3KO` |
| `description` | yes | Free-text subject description |
| `DOB`, `virus`, `injection_date`, `dilution` | no | Informational; not written to NWB |

**`experimentMetadata.csv`** — one row per experiment:

| Column | Required | Description |
| --- | --- | --- |
| `subject_id` | yes | Must match `animalList.csv` and data folder name |
| `session_description` | yes | Short session label(s); the value is passed to NWB as a single string. If your experiment has multiple sessions, separate them with `;` or `\|` for readability — keep them in the one cell. |
| `experiment_description` | yes | Manuscript Figure; Full description(s); same convention as `session_description`. |
| `keywords` | yes | Python list literal, e.g. `"['2P', 'DRC', 'pupillometry']"` |

> **Species:** both pipelines hardcode `species="Mus musculus"` when constructing the NWB `Subject`. Edit the `setSubject(...)` call in `scanimage2nwb.py` / `qcam2nwb.py` if you need a different species.

---

## Example dataset

A small test dataset covering the major code paths is available [on SharePoint](https://pitt.sharepoint.com/:f:/r/sites/TzounopoulosLab/Shared%20Documents/data/data_standardization/ophys/example_data?csf=1&web=1&e=d8XiWZ). The directory contains three subjects, with subject and experiment metadata already filled in at `example_data/animalList.csv` and `example_data/experimentMetadata.csv` (matching `subject_id` values).

| Subject | Pipelines | What it demonstrates |
| --- | --- | --- |
| `CC0001` | 2P + qcam | Full upstream data: `_tifFileList.mat`, `_moCorrROI_all.mat`, per-tif `_Pulses.mat`, `pulseLegendQcam.csv`. Exercises the priority-1 source at every fallback ladder. |
| `CC0002` | 2P + qcam | No `_Pulses.mat`, no `_tifFileList.mat`, no pulse legend in the experiment dir. The 2P pipeline needs **two** inputs and they are independent fallback ladders — completing the interactive tif-list scan does **not** generate pulse metadata. **Always required:** a pulse legend, supplied by copying one of `CC0002/pulseLegendExamples/pulseLegend2P.{csv,mat}` up one level **or** by running `python extra/matchXSG.py /path/to/CC0002` to generate `pulseLegend2P.csv` from `.xsg` timestamps (then fill in the blank `stimDelay` / `ISI` columns before running the pipeline). **Headless only:** also copy `CC0002/tifFileList_example/CC0002_tifFileList.csv` up one level — on a graphical machine the pipeline instead prompts interactively for treatment / map-file assignment. |
| `AA0337` | 2P with pupillometry | Full 2P data plus `_pulsePupilUVlegend2P_s.mat` and per-tif `_pupilFrames.mat` — exercises the pupillometry code path. The two qcamraw files have no ROI sidecar, so qcam-on-AA0337 raises an actionable `FileNotFoundError` on a headless machine; create `_qcamROI.json` per qcamraw to enable that path, or skip AA0337 in the qcam metadata. |

### Running both pipelines

Each script iterates every subject in the metadata CSV. Per-subject failures are caught and reported in a summary at the end so a single broken subject never blocks the others. Set `dataPath` to the unzipped parent directory containing the three subject folders:

```bash
# 2P pipeline → {nwb_test}/{subject}/{subject}_2P_DANDI.nwb
python scanimage2nwb.py /path/to/nwb_test \
    --subjects    example_data/animalList.csv \
    --experiments example_data/experimentMetadata.csv \
    --config      configs/params_PC.yaml

# qcam pipeline → {nwb_test}/{subject}/{subject}_qcam_DANDI.nwb
python qcam2nwb.py /path/to/nwb_test \
    --subjects    example_data/animalList.csv \
    --experiments example_data/experimentMetadata.csv \
    --config      configs/params_qcam.yaml
```

On a fresh dataset in a headless environment, the expected outcome is:

| Pipeline | Succeeds | Fails (with actionable instructions) |
| --- | --- | --- |
| 2P    | `CC0001`, `AA0337` | `CC0002` — needs `_tifFileList.csv` (template at `CC0002/tifFileList_example/`) |
| qcam  | `CC0001`, `CC0002` | `AA0337` — needs per-qcamraw `_qcamROI.json` |

On a graphical machine, the failing subjects fall through to interactive prompts (treatment / pre-post selection for 2P; ROI rectangle drawing for qcam) and convert successfully.

Validate any output:

```bash
nwbinspector /path/to/nwb_test/CC0001/CC0001_2P_DANDI.nwb --config dandi
```

### Structural test suite

The `tests/` directory contains pytest-compatible scripts (`test_2p.py`, `test_qcam.py`) that run each pipeline for `CC0001` and assert structural correctness — column presence, series counts, ROI segmentation, motion correction, fluorescence trace counts, `fileTimeInstantiate` parseability, and nwbinspector compliance under DANDI config. Point them at the unzipped dataset via the `DATA2NWB_TEST_DATA` env var:

```bash
export DATA2NWB_TEST_DATA=/path/to/nwb_test

pytest tests/                  # if pytest installed
python tests/test_qcam.py      # or run standalone
python tests/test_2p.py
```

If `DATA2NWB_TEST_DATA` is unset the tests look in a hardcoded developer location and skip cleanly when absent.

---

## 2P (ScanImage) to NWB

### Running

Experimenter and imaging parameters are configured via a YAML file in `configs/`.

- `configs/params_PC.yaml` — settings used in Cody et al 2024 (doi: 10.1523/JNEUROSCI.0939-23.2024).
- `configs/params_general.yaml` — blank template.

```bash
python scanimage2nwb.py /path/to/data

# with custom metadata and config paths
python scanimage2nwb.py /path/to/data \
    --subjects    /path/to/animalList.csv \
    --experiments /path/to/experimentMetadata.csv \
    --config      ./configs/my_params.yaml
```

Can also be run from `scanimage2nwb.ipynb`. Output: `{dataPath}/{experimentID}/{experimentID}_2P_DANDI.nwb`.

### Data directory structure

`dataPath` must be the top-level directory containing one folder per subject, named by `subject_id`.

```
dataPath/
  AA0001/                        ← experimentID / subject_id
    AA0001*.tif
    NoRMCorred/                  ← directory name set by dataProcessing.motionCorrectedTifDir
      AA0001*_NoRMCorre.tif
      AA0001_NoRMCorreParams.mat
    AA0001_moCorrROI_all.mat     (or AA0001*_roiOutput.mat)
    AA0001_tifFileList.mat       (optional — see fallbacks below)
    AA0001_tifFileList.csv       (optional — see fallbacks below)
    AA0001*_Pulses.mat           (optional — see fallbacks below)
    pulseLegend2P.mat            (optional — see fallbacks below)
    pulseLegend2P.csv            (optional — see fallbacks below)
  AA0002/
  ...
```

The motion-corrected tif subdirectory name defaults to `NoRMCorred` and is configurable via `dataProcessing.motionCorrectedTifDir` in the YAML config.

### Session start time

Session start is the earliest tif acquisition time across the experiment, parsed from each tif's ScanImage `epoch` metadata field via `lib.tifExtract.getSItifTime`. The `min(...)` over all tifs avoids an off-by-one when files are not in chronological alphabetical order.

### File sources and fallbacks

Each stage tries sources in priority order, falling back to the next if the preferred file is absent.

#### 1. Tif file list, treatment, and frame counts

Determines which `.tif` files belong to the experiment, their treatment label (e.g. `preZX1`, `postZX1`), tif type (`stim` / `map`), and frame counts.

| Priority | Source | Notes |
| --- | --- | --- |
| 1 | `{experimentID}_tifFileList.mat` | Generated by [matlabPAC_process2P](https://github.com/xiubert/matlabPAC_process2P). Contains full metadata. |
| 2 | `{experimentID}_tifFileList.csv` | Three columns: `file`, `treatment`, `type`. Frame counts read from tif metadata. Mirrors qcam's `_qcamFileList.csv`. See `example_data/tifFileList_example.csv`. |
| 3 | Interactive scan of `{experimentID}*.tif` files | Only used when a TTY or DISPLAY is available. Prompts for treatment name, pre/post assignment, and mapping file selection. Frame counts read from tif metadata. GUI dialog if a display is available; otherwise terminal prompts supporting range syntax (e.g. `1-5,7,9`). |

If none of the above are available (no MAT, no CSV, headless environment) the pipeline raises `FileNotFoundError` with instructions on creating the CSV.

Example `{experimentID}_tifFileList.csv`:

```csv
file,treatment,type
CC0002AAAA_00001_00001.tif,preZX1,stim
CC0002AAAA_00002_00001.tif,preZX1,stim
CC0002AAAA_00003_00001.tif,preZX1,map
CC0002AAAA_00004_00001.tif,postZX1,stim
CC0002AAAA_00005_00001.tif,postZX1,stim
```

#### 2. ROI masks

| Priority | Source | Notes |
| --- | --- | --- |
| 1 | `{experimentID}_moCorrROI*.mat` | Legacy MATLAB output. Single `_all.mat` → one ROI set; multiple files → one per treatment. |
| 2 | `*_roiOutput.mat` | Newer format; ROI struct at `fluo2p.roi`. Treated as equivalent to `_moCorrROI_all.mat`. |

#### 3. ROI fluorescence traces

Mean fluorescence per ROI per frame, shape `(nFrames × nROIs)` per tif.

| Priority | Source | Notes |
| --- | --- | --- |
| 1 | `{experimentID}_tifFileList.mat` | Pre-computed `moCorRawFroi` traces. |
| 2 | Motion-corrected tifs in `{motionCorrectedTifDir}/` | Computed directly from `{tif}_NoRMCorre.tif` files using ROI masks via vectorised matrix multiply. |

#### 4. Pulse / stimulus metadata

Maps sound stimulus parameters to each `.tif` file. One row per pulse/xsg (map tifs expand to one row per stimulus).

> **Strict match required:** every imaging file referenced in the pulse legend must also appear in the tif file list (section 1). The writer raises `ValueError` on the first mismatch. The reverse is not required — a tif present in the file list but absent from the pulse legend simply won't appear in the NWB stim table. If you supply a `_tifFileList.csv` covering only a subset of tifs, trim the pulse legend to that subset accordingly.

| Priority | Source | Notes |
| --- | --- | --- |
| 1 | `{experimentID}*_Pulses.mat` (one per recording) | Written automatically by [2PCI_setup](https://github.com/xiubert/2PCI_setup). Full per-pulse metadata. |
| 2 | `pulseLegend2P.mat` | Run `extra/tifPulseLegend2P.m` — requires `*_Pulses.mat` files. Aggregated struct saved as `-v7.3`. |
| 3 | `pulseLegend2P.csv` | Create manually from `example_data/pulseLegend2P_example.csv`, or generate with `extra/matchXSG.py`. `stimDelay` / `ISI` are blank when generated — fill manually. |

See [Pulse legend format](#pulse-legend-format) for column definitions.

**Generating `pulseLegend2P.mat` (MATLAB):**

```matlab
pulseLegend2P = tifPulseLegend2P('/path/to/experimentDir', true);   % scan and save
pulseLegend2P = tifPulseLegend2P('/path/to/experimentDir');          % scan only
```

**Generating `pulseLegend2P.csv` (Python):**

```bash
python extra/matchXSG.py /path/to/experiment/dir
```

Matches each `.tif` to `.xsg` files by creation timestamp. Default: each tif owns all `.xsg` files between its start time and the next tif's start time (one `.xsg` → `stim`; more → `map`, one row per `.xsg`). With `--one-per-file`, only the nearest `.xsg` at or after each tif is matched. `stimDelay` and `ISI` are not in the `.xsg` header and will be blank — fill manually if needed.

> **Note:** `lib.mat2py.getPulsesFromCSV` calls `float()` on `stimDelay` / `ISI` unconditionally, so blank values will raise. Fill them before running.

#### 5. Pupillometry (optional)

Source: `{experimentID}_pulsePupilUVlegend2P_s.mat` — pupil data as a MATLAB struct (the `_s` suffix). If your data was saved as a MATLAB table (`{experimentID}_pulsePupilUVlegend2P.mat`, no `_s`), convert it first:

```matlab
% in MATLAB, from extra/tableMAT2StructMAT.m
tableMAT2StructMAT('/path/to/experimentDir')
```

If the `_s.mat` file is absent the pupillometry section is skipped silently.

---

## Widefield (qcamraw) to NWB

Converts QImaging `.qcamraw` recordings to NWB using `OnePhotonSeries` (the correct NWB type for widefield fluorescence — carries `ImagingPlane`, indicator, excitation/emission wavelengths, etc.). One `.qcamraw` per recording; one ROI mask per treatment.

### Running

```bash
python qcam2nwb.py /path/to/data --config ./configs/params_qcam.yaml
```

Output: `{dataPath}/{experimentID}/{experimentID}_qcam_DANDI.nwb`.

### Data directory structure

```
dataPath/
  CC0001/                              ← experimentID / subject_id
    CC0001*.qcamraw
    CC0001*_qcamROI.json               (optional — one per .qcamraw, saved by interactive ROI selector)
    CC0001_qcamFileList.csv            (optional — see fallbacks below)
    CC0001*_Pulses.mat                 (optional — see fallbacks below)
    pulseLegendQcam.mat                (optional — see fallbacks below)
    pulseLegendQcam.csv                (optional — see fallbacks below)
  CC0002/
  ...
```

### File sources and fallbacks

#### 1. Frame rate

`.qcamraw` headers do not store frame rate. Set `qcam_frameRate` in the YAML config; this must match `imagingPlane_rate`. The writer raises if they disagree.

#### 2. Session start time

Read from the qcamraw header's `File_Init_Timestamp` field (QCapture format: `MM-DD-YYYY_HH:MM:SS`). Falls back to file mtime with a warning if no parseable timestamp is found. If your QCapture version uses a different header key, add `qcam_timestamp_field: YourKeyName` to the `qcam:` section of the YAML config.

> **Note on timezones:** the header timestamp carries no timezone. The implementation attaches local time. For DANDI uploads where absolute wall-clock time matters, ensure the conversion machine's timezone matches acquisition, or modify `lib/qcamraw.py:get_qcamraw_start_time`.

#### 3. File list and treatment

| Priority | Source | Notes |
| --- | --- | --- |
| 1 | `{experimentID}_qcamFileList.csv` | Columns: `file`, `treatment`, `type`. One row per `.qcamraw`. |
| 2 | `*AAZX*.qcamraw` | Checks ZX-embedded qcam filenames indicating ZX1 injection treatment. |
| 3 | `INJECTION_*_START_*.txt` | Auto-detects pre/post split from filename (e.g. `INJECTION_ZX1_START_101.txt`). |
| 4 | Glob of `{experimentID}*.qcamraw` | All files assigned `treatment='none'`, `type='stim'`. |

Example `{experimentID}_qcamFileList.csv`:

```csv
file,treatment,type
CC0001AAAA0001.qcamraw,preZX1,stim
CC0001AAAA0002.qcamraw,preZX1,stim
CC0001AAAA0003.qcamraw,postZX1,stim
CC0001AAAA0004.qcamraw,postZX1,map
```

#### 4. ROI selection

| Priority | Source | Notes |
| --- | --- | --- |
| 1 | `*response_mask*.joblib` in the experiment directory | `joblib.load()` returns a 2-D boolean numpy array matching the frame shape. Treatment-specific file (filename contains `pre`/`post` or the treatment name) takes priority over a general mask. `*contour*` files are excluded. |
| 2 | `{basename}_qcamROI.json` next to each `.qcamraw` | `{"roi": [row1, row2, col1, col2]}`, inclusive bounds. |
| 3 | Interactive matplotlib `RectangleSelector` | Drawn on the spatial dF/F map; falls back to the first frame if the recording is too short. Saves a sidecar JSON for re-runs. |

One ROI per treatment. The first `.qcamraw` of each treatment is used to resolve the ROI mask, applied to all files in that treatment. When no joblib mask is present and you want per-file rectangular ROIs, pre-populate one `_qcamROI.json` next to each `.qcamraw`.

#### 5. Pulse / stimulus metadata

> **Strict match required:** every `.qcamraw` referenced in the pulse legend must also appear in the qcam file list (section 3). The writer raises `ValueError` on the first mismatch. The reverse is not required.

| Priority | Source | Notes |
| --- | --- | --- |
| 1 | `*_Pulses.mat` (one per `.qcamraw`) | Read directly via `lib.mat2py.getPulsesPerFile`. Triggers only when every `.qcamraw` has its own companion `_Pulses.mat`. Treatments are backfilled from the qcam file list. |
| 2 | `pulseLegendQcam.mat` | Run `extra/qcamPulseLegend.m` — aggregates `*_Pulses.mat` into a single struct saved as `-v7.3`. |
| 3 | `pulseLegendQcam.csv` | Run `extra/matchXSG.py --pattern "*.qcamraw"`, or create manually. `stimDelay` / `ISI` are blank when generated — fill manually. `treatment` is auto-filled by ZX-embedded qcam filename or `INJECTION_*_START_*.txt`; otherwise left blank. |
| 4 | *(none found)* | Inventory-only stim table (one row per `.qcamraw`, NaN for `stimDelay` / `ISI`, empty strings for `pulseName` / `pulseSet` / `xsg`). Structurally valid; passes nwbinspector. |

See [Pulse legend format](#pulse-legend-format) for column definitions.

**Generating `pulseLegendQcam.mat` (MATLAB, when `*_Pulses.mat` files are present):**

```matlab
pulseLegendQcam = qcamPulseLegend('/path/to/experimentDir', true);   % scan and save
pulseLegendQcam = qcamPulseLegend('/path/to/experimentDir');          % scan only
```

**Generating `pulseLegendQcam.csv` (Python, from `.xsg` timestamps):**

```bash
python extra/matchXSG.py /path/to/experiment/dir --pattern "CC0001*.qcamraw"

# one xsg per file (always stim, no map expansion)
python extra/matchXSG.py /path/to/experiment/dir --pattern "CC0001*.qcamraw" --one-per-file
```

> **Note:** `lib.mat2py.getPulsesFromCSV` calls `float()` on `stimDelay` / `ISI` unconditionally, so blank values will raise. Fill them before running.

### Smoke-testing the writer

Before running against new data, validate with a single `.qcamraw` file:

```bash
python -m extra.qcam_smoke_test /path/to/file.qcamraw --fr 20

# also run nwbinspector against the output
python -m extra.qcam_smoke_test /path/to/file.qcamraw --fr 20 --inspect
```

Catches header-format drift between QCapture versions, pixel-ordering issues, and DANDI compliance problems before they appear in a batch run.

---

## Pulse legend format

Shared column format for both pipelines. One row per pulse/xsg. Map files appear on multiple rows (one per associated xsg). See `example_data/pulseLegend2P_example.csv` (2P) and `example_data/pulseLegendQcam_example.csv` (qcamraw) for templates.

| Column | Type | Description |
| --- | --- | --- |
| `file` | string | `.tif` or `.qcamraw` basename (imaging file key). Legacy `tif` column name is also accepted on read. |
| `type` | `stim` \| `map` | Single stimulus or BF mapping file |
| `pulseName` | string | Pulse name |
| `pulseSet` | string | Pulse set name |
| `stimDelay` | float | Delay to stimulus onset in seconds |
| `ISI` | float | Inter-stimulus interval in seconds |
| `xsg` | string | Associated `.xsg` file basename |
| `treatment` | string | Treatment label (e.g. `preZX1`, `postZX1`, `CTRL`); blank if not applicable. Legacy `condition` column is also accepted on read. |

---

## Validating and exploring NWB output

Validate against DANDI standards:

```bash
pip install -U nwbinspector
nwbinspector "{experimentID}_{2P|qcam}_DANDI.nwb" --config dandi
```

Explore interactively in the browser:

```bash
pip install --upgrade neurosift
neurosift view-nwb "{experimentID}_{2P|qcam}_DANDI.nwb"
```

---

## Related repositories

- [matlabPAC_process2P](https://github.com/xiubert/matlabPAC_process2P) — MATLAB pipeline that produces `_tifFileList.mat`, `_moCorrROI*.mat`, `_NoRMCorreParams.mat`, and contains the `meanFluoROIvt.m` GUI that the qcamraw ROI workflow mirrors.
- [2PCI_setup](https://github.com/xiubert/2PCI_setup) — ScanImage / Ephus settings for stamping per-tif pulse and pupillometry metadata.
