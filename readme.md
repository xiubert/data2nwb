# Scripts for converting 2P ophys data with pupillometry to NWB format
1. Clone repository `https://github.com/xiubert/data2nwb.git` and change to respository directory (`cd standardize2nwb`).
2. Create python venv for running these scripts to isolate dependencies: `python -m venv venvNWB`
3. Activate virtual environment:
    - unix: `source venvNWB/bin/activate`
    - Windows: double click `venvNBW/bin/activate.bat`
4. Install dependencies: `pip install -r requirements.txt`

- **Experimenter and imaging parameters are configured via YAML files in `configs/`.**
    - `configs/params_PC.yaml` contains the settings used in Cody et al 2024 (doi: 10.1523/JNEUROSCI.0939-23.2024).
    - `configs/params_general.yaml` is a blank template — copy it, fill in your experimenter and imaging settings, and pass it as the second argument to the script (see **Running** below).
- **Data directory structure:** `dataPath` should be the top-level directory containing one folder per subject, where each folder is named by `subject_id`:
    ```
    dataPath/
      AA0001/    ← subject_id
      AA0002/
      AA0003/
    ```
- **Metadata file formatting assumptions:**
    - Requires subject and experiment metadata CSVs. See `example_data/` for example files.
    - `subject_id` must match across both CSVs and the corresponding data folder name.

    **`animalList.csv`** — one row per subject, indexed by `subject_id`:

    | Column | Required | Description |
    |--------|----------|-------------|
    | `subject_id` | yes | Animal identifier; must match `experimentID` and data folder name |
    | `age` | yes | Age at time of experiment in days (integer); formatted as `P{age}D` per ISO 8601 |
    | `sex` | yes | `M` or `F` |
    | `genotype` | yes | e.g. `C57BL6/J`, `ZnT3KO` |
    | `description` | yes | Free-text subject description (e.g. virus injection details) |
    | `DOB`, `virus`, `injection_date`, `dilution` | no | Informational; not written to NWB |

    **`experimentMetadata.csv`** — one row per experiment, indexed by `subject_id`:

    | Column | Required | Description |
    |--------|----------|-------------|
    | `subject_id` | yes | Must match `subject_id` in `animalList.csv` and data folder name |
    | `session_description` | yes | Short label(s) for the session; use ` \| ` to separate multiple analyses |
    | `experiment_description` | yes | Full description(s) of the experiment; use ` \| ` to separate multiple analyses |
    | `keywords` | yes | Python list literal string, e.g. `"['2P', 'DRC', 'pupillometry']"` |

    - See https://github.com/xiubert/matlabPAC_process2P for 2P experiment data processing. Aggregated experiment metadata files correspond to this pipeline.
        - Stimuli metadata assumed to be contained in `f"{experimentID}_tifFileList.mat"`
        - ROI segmentation (masks etc) assumed to be stored in `f"{experimentID}_moCorrROI*.mat"`
        - Pupillometry metadata assumed to be contained in `f"{experimentID}_pulsePupilUVlegend2P_s.mat"` (saved as Matlab struct). For existing pupillometry tables in .mat files, a matlab script is needed to reformat table as struct to be able to load into python. See: `./extra/tableMAT2StructMAT.m`.
    - See https://github.com/xiubert/2PCI_setup for ScanImage and Ephus settings for stamping stimulus/pulse and pupillometry metadata (eg. `{tifFileName}_pulses.mat`)

- Scripts can be run within python notebook (`scanimage2nwb.ipynb`) or from the command line:

**Running:**
```bash
# all defaults (./data/animalList.csv, ./data/experimentMetadata.csv, ./configs/params_PC.yaml)
python scanimage2nwb.py /path/to/data

# custom paths
python scanimage2nwb.py /path/to/data \
    --subjects /path/to/animalList.csv \
    --experiments /path/to/experimentMetadata.csv \
    --config ./configs/my_params.yaml
```
Scripts output `.nwb` file that should conform to DANDI data standards (see below for validation).

- Explore NWB output file with `neurosift`. Running will open web browser view of `.nwb` file.
    1. in python env: `pip install --upgrade neurosift`
    2. `neurosift view-nwb AA0304_DANDI.nwb`
- After conversion, confirm NWB format conforms to DANDI standards w/ nwbinspector:
    1. in python env: `pip install -U nwbinspector`
    2. `nwbinspector "AA0304.nwb" --config dandi`

