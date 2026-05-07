"""
Helpers for pulling metadata from .mat files in ScanImage 2P experiment.
"""

import csv
import h5py
import numpy as np
from scipy.io import loadmat
import os
from pathlib import Path
import lib.tifExtract

try:
    import tkinter as tk
    from tkinter import simpledialog
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False


def getH5stringList(h5references,h5file):
    """
    Helper for grabbing a list of strings from list of h5 references.

    Args:
        h5references (list[h5reference]): each element an h5 reference object
        h5file (h5file object)
    Returns:
        string_list (list[str]): each element a string from h5reference
    """
    string_list = []
    for r in h5references:
        string_list.append("".join(chr(c.item()) for c in h5file[r][:]))
    return string_list


def getMatCellArrayOfStr(matPath: str, varPath: list[str]) -> list[str]:
    """
    Returns list of strings from cell array of strings in matlab.

    Args:
        matPath (str): file path of .mat file
        varPath (list[str]): nested path of cell array in mat file
    Returns: 
        string_list (list[str]): each element a string from h5reference
    """
    with h5py.File(matPath, "r") as h5:
        h5_ref = h5
        for key in varPath:
            h5_ref = h5_ref[key]

        references = h5_ref[0]
        string_list = getH5stringList(references,h5)
    return string_list


def getMatCellArrayOfNum(matPath: str, varPath: list[str]) -> list[float]:
    """
    Returns list of nums from cell array of num in matlab.

    Args:
        matPath (str): file path of .mat file
        varPath (list[str]): nested path of cell array in mat file

    Returns:
        numList (list[float]): each element a float from h5reference
    """
    with h5py.File(matPath, "r") as h5:
        h5_ref = h5
        for key in varPath:
            h5_ref = h5_ref[key]

        numList = []
        references = h5_ref[0]
        for r in references:
            numList.append(np.array(h5[r])[0][0])
        return numList
    

def getMoCorrShiftParams(moCorrMatPath: str, nFrames: list[int] = None, concatenate: bool = False) -> np.ndarray:
    """
    Returns numpy array of xy translation shifts of raw .tif images.

    Args:
        moCorrMatPath (str): file path of motion correction parameters file eg AA0304_NoRMCorreParams.mat
        nFrames (list[int]): each element a frame count for associated .tif file
        concatenate (bool): whether or not to return shifts in one concatenated array

    Returns:
        allshifts (list): each element is shifts for associated .tif file
        params: motion correction parameters

    """
    moCorrData = loadmat(moCorrMatPath)

    for i,cond in enumerate(moCorrData['NoRMCorreParams'].dtype.fields):
        if i==0:
            # unnest
            shifts = np.stack(moCorrData['NoRMCorreParams'][cond][0][0]['shifts'][0][0]['shifts'].squeeze()).squeeze()
        else:
            shifts = np.append(shifts,np.stack(moCorrData['NoRMCorreParams'][cond][0][0]['shifts'][0][0]['shifts'].squeeze()).squeeze(),
                            axis=0)
            
    params = moCorrData['NoRMCorreParams'][cond][0][0]['options_nonrigid'][0]

    if concatenate==True:
        return shifts
    
    cumframes = np.cumsum(nFrames)
    firstShift = cumframes-nFrames
    lastShift = cumframes-1

    allshifts = []
    for start,end in zip(firstShift,lastShift):
        allshifts.append(shifts[start:end+1])

    return allshifts,params


def getROImasks(roiMatPath: str) -> np.ndarray:
    """
    Returns 1xNumberOfROI array of ROI image masks provided path to experiment .mat file containing
    ROIs drawn on motion corrected data.

    Handles two file conventions:
      - *_moCorrROI*.mat : format via motion correction in processAnimal2P.m, struct at 'moCorROI' (scipy loadmat)
      - *_roiOutput.mat  : format via roiGUI, struct at fluo2p.roi (h5py)

    Args:
        roiMatPath (str): file path to .mat containing ROI segmentation data
    Returns:
        IDs (np.array): numpy array of int corresponding to index of ROI
        masks (np.array): numpy array of ROI masks
    """
    roiData = loadmat(roiMatPath)
    if os.path.basename(roiMatPath).endswith('_roiOutput.mat'):
        # fluo2p is (1,1) struct; roi field is a (1,N) struct array of ROIs
        roi = roiData['fluo2p'][0, 0]['roi']
        IDs = np.array([int(r.flat[0]) for r in roi['ID'].flat])
        masks = np.array([m for m in roi['mask'].flat])
    else:
        masks = roiData['moCorROI'][0]['mask']
        IDs = np.concatenate(roiData['moCorROI'][0]['ID']).astype(int)

    return IDs, masks


def getROIfluo(fluoMatPath: str, 
               concatenate: bool = False) -> tuple[list[str],list[np.ndarray]]:
    """
    Extract motion corrected fluorescence traces from tifFileList.
    Returns corresponding tif, allFrames X ROI

    Args:
        fluoMatPath (str): file path to .mat containing ROI fluorescence traces eg AA0304_tifFileList.mat
        concatenate (bool): whether or not to concatenate all responses across time/frames
    
    Returns:
        tifs (list):each element a tif file names associated with each response trace
        responses: list of numpy arrays of fluorescence responses for each ROI
    """
    with h5py.File(fluoMatPath, "r") as h5:
        tifTypes = list(h5['tifFileList'].keys())

        tifs,responses = [],[]
        for tifType in tifTypes:
            tifNameRefs = h5['tifFileList'][tifType]['name'][0]
            tifs.extend(lib.mat2py.getH5stringList(tifNameRefs,h5))
        
            responseRefs = h5['tifFileList'][tifType]['moCorRawFroi'][0]
            arr = (np.array(h5[responseRefs[0]]) if concatenate else [np.array(h5[responseRefs[0]])])
            for ref in responseRefs[1:]:
                if concatenate==True:
                    arr = np.append(arr,np.array(h5[ref]),0)
                else:
                    arr.append(np.array(h5[ref]))
            responses.extend(arr)

    return tifs,responses


def getROIfluoFromTifs(tifList: list[str],
                       masks: np.ndarray,
                       tifDir: str,
                       tifSuffix: str = '_NoRMCorre.tif',
                       concatenate: bool = False) -> tuple[list[str], list[np.ndarray]]:
    """
    Fallback for getROIfluo when no tifFileList.mat is present.
    Computes mean ROI fluorescence directly from motion-corrected tif files.

    For each tif, loads image data (nFrames x H x W) and computes the mean pixel
    value within each ROI mask per frame, yielding an (nFrames x nROIs) array.

    Args:
        tifList (list[str]): tif filenames (base names, as returned by getTifList)
        masks (np.ndarray): (nROIs, H, W) uint8 ROI mask array from getROImasks
        tifDir (str): directory containing motion-corrected tif files
        tifSuffix (str): suffix appended to base tif name for motion-corrected file,
                         e.g. '_NoRMCorre.tif'
        concatenate (bool): if True return a single concatenated (totalFrames x nROIs)
                            array instead of a list

    Returns:
        tifs (list[str]): tif filenames, matching tifList order
        responses (list[np.ndarray] | np.ndarray): each element (nFrames x nROIs),
            or single concatenated array when concatenate=True
    """
    mask_bool = (masks > 0).reshape(len(masks), -1).T.astype(np.float32)  # (H*W, nROIs)
    mask_sums = mask_bool.sum(axis=0)  # (nROIs,)

    tifs, responses = [], []
    for tif in tifList:
        moCorrTif = os.path.join(tifDir, tif.replace('.tif', tifSuffix))
        imgData = lib.tifExtract.getSItifData(moCorrTif, getMetadata=False)
        nF = imgData.shape[0]
        img_flat = imgData.reshape(nF, -1).astype(np.float32)  # (nFrames, H*W)
        response = (img_flat @ mask_bool) / mask_sums             # (nFrames, nROIs)
        tifs.append(tif)
        responses.append(response)
        print(f"  {tif}: {nF} frames, {len(masks)} ROIs")

    if concatenate:
        return tifs, np.concatenate(responses, axis=0)
    return tifs, responses


def getPupilDataProcessed(pupilMatPath: str, pupilFrameRate: float = 10):
    """
    Helper to grab relevant pupil metadata from experiment dir. 

    Args:
        pupilMatPath (str): file path of .mat containing pupil data eg. AA0304_pulsePupilUVlegend2P_s.mat (pupil data stored as struct in .mat)
        pupilFrameRate (float): frame rate of pupil video
    Returns:
        pupilDataProcessed (dict): pupillometry metadata
    """
    getNestedStructData = lambda x: x[0][0]
    pupilMatData = loadmat(pupilMatPath)

    pupilDataProcessed = {
        'pupilFrameFiles': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['pupilFrameFile'])),
        'pupilRadius': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['pupilRad'])),
        'DeepLabCutModel': list(map(getNestedStructData,pupilMatData['pulsePupilLegend2P'][:]['model'])),
        'frameRate': pupilFrameRate
    }

    return pupilDataProcessed


def getPupilImg(pupilMatPath: str) -> np.ndarray:
    """
    Helper to grab pupil img data from save .mat file containing img frames.

    Args:
        pupilMatPath (str): file path of .mat containing pupil data eg. AA0304_pulsePupilUVlegend2P_s.mat (pupil data stored as struct in .mat)
    Returns:
        np.array of pupil video data (time) x (X) x (Y)
    """
    pupilImgData = loadmat(pupilMatPath)['pupilFrames']

    return np.transpose(pupilImgData, (2, 0, 1))


def getPulses(file_path: str, fileType: str) -> dict:
    """
    Grabs stimulation metadata associated with the provided imaging file.

    Reads `<basename>_Pulses.mat` next to the imaging file. Works for any
    imaging file extension (.tif, .qcamraw) since the lookup uses splitext.

    Args:
        file_path (str): file path to imaging file (.tif or .qcamraw)
        fileType (str): 'stim' (single pulse) or 'map' (multiple pulses, BF mapping)
    Returns:
        pulseParams (dict): parameters of delivered pulse stimuli
        pulse (dict): metadata associated with pulse/stimuli eg pulse name etc
    """
    base, _ = os.path.splitext(file_path)
    mat = loadmat(base + '_Pulses.mat')

    pulseParams,pulse = {},{}
    pulseParams['traceAcqTime'] = lib.tifExtract.parse_datetime_list(mat['params'][0]['acquisitionStartTime'][0][0])
    pulseParams['stimDelay'] = mat['params'][0]['stimDelay'][0][0][0]
    pulseParams['ISI'] = mat['params'][0]['ISI'][0][0][0]

    if fileType=='stim':
        pulse['pulseSet'] = mat['pulse'][0]['pulseset'][0][0]
        pulse['pulseName'] = mat['pulse'][0]['pulsename'][0][0]
        pulse['xsg'] = mat['pulse'][0]['curXSG'][0][0].split('\\')[-1]
    elif fileType=='map':
        pulse['pulseSet'] = np.concatenate(np.squeeze(mat['pulse'][0]['pulseset']))
        pulse['pulseName'] = np.concatenate(np.squeeze(mat['pulse'][0]['pulsename']))
        pulse['xsg'] = list(map(lambda x: x.split('\\')[-1],np.concatenate(np.squeeze(mat['pulse'][0]['curXSG']))))
    
    return pulseParams,pulse


def _select_one(title: str, prompt: str, items: list[str], base_dir: str = '') -> str:
    """
    Prompt user to select a single item from a list.
    Uses tkinter GUI when a display is available, otherwise terminal.

    Args:
        title (str): window title / section header
        prompt (str): label above the list
        items (list[str]): filenames to choose from
        base_dir (str): prepended to the chosen filename to return a full path

    Returns:
        str: full path to selected file
    """
    if _TK_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()

            dlg = tk.Toplevel(root)
            dlg.title(title)
            dlg.grab_set()

            tk.Label(dlg, text=prompt, justify=tk.LEFT).pack(padx=8, pady=(8, 2), anchor='w')

            frame = tk.Frame(dlg)
            frame.pack(padx=8, pady=4, fill=tk.BOTH, expand=True)

            scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
            listbox = tk.Listbox(frame, selectmode=tk.SINGLE, yscrollcommand=scrollbar.set,
                                 width=60, height=min(len(items) + 1, 20))
            scrollbar.config(command=listbox.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            for item in items:
                listbox.insert(tk.END, item)
            listbox.selection_set(0)

            chosen = [None]

            def on_ok():
                sel = listbox.curselection()
                if sel:
                    chosen[0] = items[sel[0]]
                dlg.destroy()

            btn_frame = tk.Frame(dlg)
            btn_frame.pack(pady=6)
            tk.Button(btn_frame, text='OK', width=10, command=on_ok).pack()

            dlg.wait_window()
            root.destroy()

            if chosen[0] is not None:
                return os.path.join(base_dir, chosen[0])
        except tk.TclError:
            print("No display available — using terminal selection.")

    # terminal fallback
    print(f"\n{title}\n{prompt}")
    for i, name in enumerate(items):
        print(f"  [{i+1:>2}] {name}")
    while True:
        raw = input("  Select one (number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return os.path.join(base_dir, items[int(raw) - 1])
        print(f"  Please enter a number between 1 and {len(items)}.")


def _parse_range_selection(s: str, n: int) -> list[int]:
    """
    Parse a selection string such as "1-5,7,9" into 0-based indices.
    Input uses 1-based numbering. Returns empty list for blank input.

    Args:
        s (str): user input string, e.g. "1-3,5,7-9"
        n (int): total number of items (used for bounds checking)

    Returns:
        list[int]: sorted unique 0-based indices
    """
    indices = set()
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            indices.update(range(int(start) - 1, int(end)))
        else:
            indices.add(int(part) - 1)
    return sorted(i for i in indices if 0 <= i < n)


def _listdlg_terminal(title: str, prompt: str, items: list[str], likely_selected: list[int] = None) -> list[int]:
    """
    Terminal fallback for _listdlg. Prints a numbered list and reads a range/index selection.

    Args:
        title (str): section header printed above the prompt
        prompt (str): label printed above the list
        items (list[str]): items to display
        likely_selected (list[int]): 0-based indices flagged as likely matches (printed with *)

    Returns:
        list[int]: 0-based indices of selected items, empty if blank/cancelled
    """
    print(f"\n{title}")
    print(f"{prompt}")
    for i, name in enumerate(items):
        hint = ' *' if likely_selected and i in likely_selected else ''
        print(f"  [{i+1:>2}] {name}{hint}")
    if likely_selected:
        print("       (* likely map file based on size)")
    raw = input("  Selection (e.g. 1-5,7,9 — blank for none): ").strip()
    if not raw:
        return []
    return _parse_range_selection(raw, len(items))


def _listdlg(title: str, prompt: str, items: list[str], likely_selected: list[int] = None) -> list[int]:
    """
    Multi-select dialog. Uses tkinter GUI when a display is available, otherwise
    falls back to a terminal numbered-list prompt.

    Args:
        title (str): window title (GUI) / section header (terminal)
        prompt (str): label text above the list
        items (list[str]): items to display
        likely_selected (list[int]): 0-based indices to visually hint as likely matches

    Returns:
        list[int]: 0-based indices of selected items, empty if cancelled
    """
    if _TK_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()

            dlg = tk.Toplevel(root)
            dlg.title(title)
            dlg.grab_set()

            tk.Label(dlg, text=prompt, justify=tk.LEFT).pack(padx=8, pady=(8, 2), anchor='w')

            frame = tk.Frame(dlg)
            frame.pack(padx=8, pady=4, fill=tk.BOTH, expand=True)

            scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
            listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set,
                                 width=60, height=min(len(items) + 1, 20))
            scrollbar.config(command=listbox.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            for item in items:
                listbox.insert(tk.END, item)

            if likely_selected:
                for i in likely_selected:
                    listbox.itemconfig(i, {'bg': '#fffacd'})

            selected_indices = []

            def on_ok():
                selected_indices.extend(listbox.curselection())
                dlg.destroy()

            def on_cancel():
                dlg.destroy()

            btn_frame = tk.Frame(dlg)
            btn_frame.pack(pady=6)
            tk.Button(btn_frame, text='OK', width=10, command=on_ok).pack(side=tk.LEFT, padx=4)
            tk.Button(btn_frame, text='Cancel', width=10, command=on_cancel).pack(side=tk.LEFT, padx=4)

            dlg.wait_window()
            root.destroy()
            return selected_indices

        except tk.TclError:
            print("No display available — using terminal selection.")

    return _listdlg_terminal(title, prompt, items, likely_selected)


def getTifListFromDir(dataPath: str, experimentID: str) -> tuple:
    """
    Interactive fallback for getTifList when no MATLAB tifFileList.mat is present.

    Lists .tif files found in the experiment directory, then opens GUI dialogs to
    collect treatment name, pre/post assignment, and mapping file selection.
    Frame counts are read directly from each .tif via tifExtract.

    Args:
        dataPath (str): parent folder path containing experimentID subfolder
        experimentID (str): experiment id (usually parent folder name eg AA0304)

    Returns:
        tifList (list[str]): each element a .tif filename (relative, matching mat2py convention)
        tifTypeList (list[str]): 'stim' or 'map' for each tif
        treatments (list[str]): treatment label for each tif
        nFrames (list[int]): frame count for each tif
    """
    exp_dir = Path(dataPath) / experimentID
    tif_paths = sorted(exp_dir.glob(f"{experimentID}*.tif"))

    if not tif_paths:
        print(f"No .tif files found in {exp_dir}")
        return [], [], [], []

    tif_names = [p.name for p in tif_paths]
    print(f"Found {len(tif_names)} .tif file(s):")
    for name in tif_names:
        print(f"  {name}")

    # --- treatment assignment ---
    if _TK_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()
            treatment_name = simpledialog.askstring(
                'Treatment Input',
                'Enter treatment name (cancel if none):',
                initialvalue='ZX1',
                parent=root
            )
            root.destroy()
        except tk.TclError:
            print("No display available — using terminal input.")
            treatment_name = input("Enter treatment name (blank for none) [ZX1]: ").strip() or None
    else:
        treatment_name = input("Enter treatment name (blank for none) [ZX1]: ").strip() or None

    treatment = ['none'] * len(tif_names)

    if treatment_name:
        pre_label = f'pre{treatment_name}'
        post_label = f'post{treatment_name}'

        pre_idx = _listdlg(
            title='Pre-treatment selection',
            prompt=f'Select PRE-{treatment_name} tif files:',
            items=tif_names
        )
        pre_set = set(pre_idx)
        treatment = [
            pre_label if i in pre_set else post_label
            for i in range(len(tif_names))
        ]

    # --- mapping file selection ---
    likely_map = [
        i for i, p in enumerate(tif_paths)
        if p.stat().st_size > 11_000_000
    ]
    if likely_map:
        print("Likely map files (>11 MB):")
        for i in likely_map:
            print(f"  {tif_names[i]}")

    map_idx = _listdlg(
        title='BF mapping file selection',
        prompt='Select tif files for BF mapping (cancel if none):',
        items=tif_names,
        likely_selected=likely_map
    )

    tif_type_list = ['stim'] * len(tif_names)
    for i in map_idx:
        tif_type_list[i] = 'map'
        treatment[i] = (treatment[i] + ' FRAmap').strip()

    # --- frame counts ---
    print("Reading frame counts from tif metadata...")
    n_frames = []
    for p in tif_paths:
        _, _, nf, _ = lib.tifExtract.getSItifData(str(p), getMetadata=True)
        n_frames.append(int(nf))
        print(f"  {p.name}: {int(nf)} frames")

    return tif_names, tif_type_list, treatment, n_frames


def getTifTypes(tifFileListMatPath: str) -> list[str]:
    """
    Returns available tif types from tifFileList mat file.
    
    Looks for structure fields in tifFileList variable within *_tifFileList.mat mat file.
    matlabPAC_process2P analysis pipeline groups tif files by FRA mapping or stim in general. 

    Args:
        tifFileListMatPath (str): filepath to .mat containing metadata of .tif files eg AA0304_tifFileList.mat
    Returns:
        list with each element indicated whether tif has single pulse/stimulus or multiple (in case of BF mapping .tif)
    """
    with h5py.File(tifFileListMatPath, "r") as h5:
        return list(h5['tifFileList'].keys())
    

def getTifList(dataPath: str, experimentID: str, tifListMatFilename: str | None):
    """
    Gets tif list, frame counts, and associated treatment from tifFileList.mat file.
    Returns lists of length n tifs.

    Args:
        dataPath (str): parent folder path to tifFileList.mat
        experimentID: experiment id (usually parent folder name eg AA0304)
        tifListMatFilename: filename of tifFileList.mat eg AA0304_tifFileList.mat
    Returns:
        tifList (list[str]): each element a .tif filename
        tifTypeList (list[str]): each element indicates .tif type (single pulse or multiple pulses/BFmap)
        treatments (list[str]): element indicates treatment associated w/ .tif
        nFrames (list[int]): element indicates frame count for associated .tif
    """
    tifListMatPath = os.path.join(dataPath, experimentID, tifListMatFilename)
    if os.path.exists(tifListMatPath):
        print("Found tifFileList generated from MATLAB analysis.")
        tifTypes = getTifTypes(tifListMatPath)

        tifList, tifTypeList, nFrames, treatments = [], [], [], []

        for tifType in tifTypes:
            print(f"recording includes: {tifType}")
            tifs = getMatCellArrayOfStr(tifListMatPath,varPath = ['tifFileList',tifType,'name'])
            treatment = getMatCellArrayOfStr(tifListMatPath,varPath = ['tifFileList',tifType,'treatment'])
            nFrame = getMatCellArrayOfNum(tifListMatPath,varPath = ['tifFileList',tifType,'nFrames'])
            if tifType=='map':
                treatment = [t.replace('BFmap','').replace("FRAmap","").strip() for t in treatment]
            tifList.extend(tifs)
            tifTypeList.extend([tifType]*len(tifs))
            nFrames.extend(nFrame)
            treatments.extend(treatment)
    else:
        print("tifFileList not found. Falling back to interactive directory scan...")
        tifList, tifTypeList, treatments, nFrames = getTifListFromDir(dataPath, experimentID)

    return tifList, tifTypeList, treatments, nFrames


def getPulsesPerFile(experimentDir: str, fileList: list[str], fileTypeList: list[str]):
    """
    Reads stimulus metadata from `*_Pulses.mat` companions of imaging files.

    Generalised over imaging file format — works for both ScanImage `.tif`
    and QImaging `.qcamraw` files (or any imaging file with a sibling
    `<basename>_Pulses.mat`).

    Args:
        experimentDir (str): directory containing the imaging files and
            their `*_Pulses.mat` companions.
        fileList (list[str]): each element a basename of an imaging file
        fileTypeList (list[str]): each element 'stim' (single pulse) or
            'map' (multiple pulses / BF mapping)

    Returns:
        Tuple of 8 lists (one row per pulse/xsg; map files expand to one
        row per associated xsg):
            files, fileTypes, stimDelays, ISIs, pulseNames, pulseSets,
            xsg, treatments
        `treatments` is filled with empty strings; populate it from a
        higher-level file list if needed.
    """
    files, fileTypes = [],[]
    stimDelays, ISIs, pulseNames, pulseSets, xsg, treatments = [],[],[],[],[],[]

    for fname,fileType in zip(fileList,fileTypeList):
        pulseParams,pulse = getPulses(os.path.join(experimentDir,fname),fileType)
        pulseSet,pulseName,x = list(pulse.values())

        if fileType=='map':
            fname,fileType,ISI,stimDelay = zip(*[(fname,
                                   fileType,
                                   pulseParams['stimDelay'],
                                   pulseParams['ISI']) for _ in pulseName])
            xsg.extend(x)
            pulseNames.extend(pulseName)
            pulseSets.extend(pulseSet)
            ISIs.extend(ISI)
            stimDelays.extend(stimDelay)
            files.extend(fname)
            fileTypes.extend(fileType)
            treatments.extend(['' for _ in pulseName])

        else:
            files.append(fname)
            fileTypes.append(fileType)
            xsg.append(x)
            pulseNames.append(pulseName)
            pulseSets.append((np.unique(pulseSet)[0] if len(np.unique(pulseSet))==1 else pulseSet))
            ISIs.append(pulseParams['ISI'])
            stimDelays.append(pulseParams['stimDelay'])
            treatments.append('')

    return files, fileTypes, stimDelays, ISIs, pulseNames, pulseSets, xsg, treatments


def getTifPulses(dataPath: str, experimentID: str, tifList: list[str], tifTypeList: list[str]):
    """
    Backward-compatible wrapper around getPulsesPerFile for 2P call sites.
    """
    return getPulsesPerFile(os.path.join(dataPath, experimentID), tifList, tifTypeList)


def getPulsesFromLegend(legendMatPath: str) -> tuple:
    """
    Reads pulse metadata from a pulseLegend2P.mat or pulseLegendQcam.mat file.

    pulseLegend2P.mat is generated by tifPulseLegend2P.m and uses a 'tif' field.
    pulseLegendQcam.mat is generated by qcamPulseLegend.m and uses a 'file' field.

    Args:
        legendMatPath (str): path to the legend .mat file (saved as -v7.3)

    Returns:
        files, fileTypes, stimDelays, ISIs, pulseNames, pulseSets, xsg
        matching the format returned by getTifPulses
    """
    def _h5str(obj):
        """Read a single HDF5 char array (uint16) as a Python string."""
        return ''.join(chr(c.item()) for c in obj[:].flat)

    def _basename(path_str):
        return os.path.basename(path_str.replace('\\', '/'))

    def _resolve_strings(ref, h5):
        """Return list of strings from an h5 reference: cell array → multiple, char array → one."""
        obj = h5[ref]
        if obj.dtype.kind == 'O':
            return getH5stringList(obj[()].flatten(), h5)
        return [_h5str(obj)]

    with h5py.File(legendMatPath, 'r') as h5:
        legend = h5['pulseLegendQcam'] if 'pulseLegendQcam' in h5 else h5['pulseLegend2P']
        file_key = 'file' if 'file' in legend else 'tif'

        tifs       = getH5stringList(legend[file_key][0], h5)
        stimDelays = [float(np.array(h5[r]).flat[0]) for r in legend['stimDelay'][0]]
        ISIs       = [float(np.array(h5[r]).flat[0]) for r in legend['ISI'][0]]

        # read treatment field; fall back to condition for old MAT files
        try:
            treatments_raw = getH5stringList(legend['treatment'][0], h5)
        except (KeyError, Exception):
            try:
                treatments_raw = getH5stringList(legend['condition'][0], h5)
            except (KeyError, Exception):
                treatments_raw = [''] * len(tifs)

        # expand map entries to one row per pulse/xsg, matching getTifPulses behaviour
        out_tifs, out_types, out_delays, out_isis = [], [], [], []
        out_names, out_sets, out_xsg, out_treatments = [], [], [], []

        for tif, stimDelay, ISI, treat, name_ref, set_ref, xsg_ref in zip(
                tifs, stimDelays, ISIs, treatments_raw,
                legend['pulseName'][0], legend['pulseSet'][0], legend['xsg'][0]):

            xsg_strings   = _resolve_strings(xsg_ref,  h5)
            pulse_names   = _resolve_strings(name_ref, h5)
            pulse_sets    = _resolve_strings(set_ref,  h5)

            n = len(xsg_strings)
            tif_type = 'map' if n > 1 else 'stim'

            # pad name/set lists to n in case MATLAB stored only one value for map
            pulse_names = (pulse_names * n)[:n]
            pulse_sets  = (pulse_sets  * n)[:n]

            out_tifs.extend([tif] * n)
            out_types.extend([tif_type] * n)
            out_delays.extend([stimDelay] * n)
            out_isis.extend([ISI] * n)
            out_names.extend(pulse_names)
            out_sets.extend(pulse_sets)
            out_xsg.extend([_basename(s) for s in xsg_strings])
            out_treatments.extend([treat] * n)

    return out_tifs, out_types, out_delays, out_isis, out_names, out_sets, out_xsg, out_treatments


def getPulsesFromCSV(csvPath: str) -> tuple:
    """
    Reads pulse metadata from a pulseLegend2P.csv or pulseLegendQcam.csv file.
    Fallback for getTifPulses / getPulsesFromLegend when no .mat files are available.

    Expected columns (see example_data/pulseLegend2P_example.csv):
        tif, type, pulseName, pulseSet, stimDelay, ISI, xsg

    One row per pulse/xsg — map tifs appear once per associated xsg file.

    Args:
        csvPath (str): path to the CSV file

    Returns:
        tifs, tifTypes, stimDelays, ISIs, pulseNames, pulseSets, xsg
        matching the format returned by getTifPulses
    """
    files, fileTypes, stimDelays, ISIs, pulseNames, pulseSets, xsg, treatments = [], [], [], [], [], [], [], []

    with open(csvPath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # imaging file key column: prefer 'file', fall back to legacy 'tif'
            if 'file' in row:
                files.append(row['file'])
            elif 'tif' in row:
                files.append(row['tif'])
            else:
                raise KeyError(
                    f"{csvPath}: pulse legend must have a 'file' (or legacy 'tif') column.")
            fileTypes.append(row['type'])
            stimDelays.append(float(row['stimDelay']))
            ISIs.append(float(row['ISI']))
            pulseNames.append(row['pulseName'])
            pulseSets.append(row['pulseSet'])
            xsg.append(row['xsg'])
            # read 'treatment'; fall back to 'condition' for old CSVs
            treatments.append(row.get('treatment', row.get('condition', '')))

    return files, fileTypes, stimDelays, ISIs, pulseNames, pulseSets, xsg, treatments


