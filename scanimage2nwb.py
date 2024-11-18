import pandas as pd
import ast
import os
import lib.nwbScanImage
import sys

# Access path argument
if len(sys.argv) > 1:
    print(f"dataPath: {sys.argv[1]}")
else:
    raise("No path provided.")
dataPath = sys.argv[1]

# retrieve subject and experiment metadata from log tables
subjects = pd.read_csv('./data/animalList.csv')
subjects = subjects.set_index('subject_id')
experiments = pd.read_csv('./data/experimentMetadata.csv')
experiments = experiments.set_index('experimentID')
experiments['keywords'] = experiments['keywords'].apply(ast.literal_eval)

for experimentID,d in experiments.iterrows():
    print(f"processing: {experimentID}...")

    outputNWBpath = os.path.join(dataPath,experimentID,f"{experimentID}_DANDI.nwb")
    
    subject = lib.nwbScanImage.setSubject(
        subject_id=experimentID,
        age=f"P{subjects.loc[experimentID]['age']}D",
        species="Mus musculus",
        sex=subjects.loc[experimentID]['sex'],
        genotype=subjects.loc[experimentID]['genotype'],
        description=subjects.loc[experimentID]['description']
    )

    lib.nwbScanImage.genNWBfromScanImage_pc(
        experimentID=experimentID, 
        dataPath=dataPath, 
        NWBoutputPath=outputNWBpath,
        subject=subject,
        session_description=d['session_description'],
        experiment_description=d['experiment_description'],
        keywords=d['keywords'],
        **lib.nwbScanImage.PARAMS_nwbFilePC,
        **lib.nwbScanImage.PARAMS_imagingPC
        )