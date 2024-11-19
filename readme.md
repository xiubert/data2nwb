# Scripts for converting 2P ophys data with pupillometry to NWB format
1. Clone repository `https://github.com/xiubert/standardize2nwb.git` and change to respository directory (`cd standardize2nwb`).
2. Create python venv for running these scripts to isolate dependencies: `python -m venv venvNWB`
3. Activate virtual environment:
    - unix: `source venvNWB/bin/activate`
    - Windows: double click `venvNBW/bin/activate.bat`
4. Install dependencies: `pip install -r requirements.txt`

- **Scripts are currently specific to PC data in Cody et al 2024 (doi: 10.1523/JNEUROSCI.0939-23.2024), but can be abstracted to other experiments.**
    - **At the very least `PARAMS_*` variables in `data2nwb/lib/nwbScanImage.py` will need to be updated for corresponding experimenter and imaging settings.**
- **Metadata file formatting assumptions:**
    - Requires subject and experiment metadata in `./data/animalList.csv` and `./data/experimentMetadata.csv`. See `scanimage2nwb.ipynb` for required columns and formatting.
    - See https://github.com/xiubert/matlabPAC_process2P for 2P experiment data processing. Aggregated experiment metadata files correspond to this pipeline.
        - Stimuli metadata assumed to be contained in `f"{experimentID}_tifFileList.mat"`
        - ROI segmentation (masks etc) assumed to be stored in `f"{experimentID}_moCorrROI*.mat"`
        - Pupillometry metadata assumed to be contained in `f"{experimentID}_pulsePupilUVlegend2P_s.mat"` (saved as Matlab struct). For existing pupillometry tables in .mat files, a matlab script is needed to reformat table as struct to be able to load into python. See: `./extra/tableMAT2StructMAT.m`.
    - See https://github.com/xiubert/matlabPAC_2Pacquisition for ScanImage and Ephus settings for stamping stimulus/pulse and pupillometry metadata (eg. `{tifFileName}_pulses.mat`)

   
- Subject_ID is assumed to be same as experiment parent folder (`subject_id==experimentID`)

- Scripts can be run within python notebook (`scanimage2nwb.ipynb` or from command line `scanimage2nwb.py`)
    - Scripts output `.nwb` file that should conform to DANDI data standards (see below for validation)

- Explore NWB output file with `neurosift`. Running will open web browser view of `.nwb` file.
    1. in python env: `pip install --upgrade neurosift`
    2. `neurosift view-nwb AA0304_DANDI.nwb`
- After conversion, confirm NWB format conforms to DANDI standards w/ nwbinspector:
    1. in python env: `pip install -U nwbinspector`
    2. `nwbinspector "AA0304.nwb" --config dandi`

