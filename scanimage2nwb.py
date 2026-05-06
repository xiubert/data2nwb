import pandas as pd
import ast
import os
import yaml
import argparse
import lib.nwbScanImage

parser = argparse.ArgumentParser(
    description="Convert ScanImage 2P data to NWB format."
)
parser.add_argument(
    'dataPath',
    help='Top-level directory containing per-subject data folders (each folder named by subject_id).'
)
parser.add_argument(
    '--subjects',
    default='./data/animalList.csv',
    metavar='PATH',
    help='Path to animalList.csv (default: ./data/animalList.csv).'
)
parser.add_argument(
    '--experiments',
    default='./data/experimentMetadata.csv',
    metavar='PATH',
    help='Path to experimentMetadata.csv (default: ./data/experimentMetadata.csv).'
)
parser.add_argument(
    '--config',
    default='./configs/params_PC.yaml',
    metavar='PATH',
    help='Path to YAML config file (default: ./configs/params_PC.yaml).'
)
args = parser.parse_args()

dataPath = args.dataPath
print(f"dataPath:    {dataPath}")
print(f"subjects:    {args.subjects}")
print(f"experiments: {args.experiments}")
print(f"config:      {args.config}")

with open(args.config) as f:
    cfg = yaml.safe_load(f)

# retrieve subject and experiment metadata from log tables
subjects = pd.read_csv(args.subjects)
subjects = subjects.set_index('subject_id')
experiments = pd.read_csv(args.experiments)
experiments = experiments.set_index('subject_id')
experiments['keywords'] = experiments['keywords'].apply(ast.literal_eval)

for experimentID,d in experiments.iterrows():
    print(f"processing: {experimentID}...")

    outputNWBpath = os.path.join(dataPath,experimentID,f"{experimentID}_2P_DANDI.nwb")

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
        **cfg['nwb_file'],
        **cfg['imaging']
        )