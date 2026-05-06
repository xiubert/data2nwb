"""
Convert QImaging .qcamraw widefield epifluorescence data to NWB.

Mirrors scanimage2nwb.py — same CSV inputs, same YAML config layout,
output written to {dataPath}/{experimentID}/{experimentID}_qcam_DANDI.nwb.
"""
import argparse
import ast
import os

import pandas as pd
import yaml

import lib.nwbScanImage   # for setSubject
import lib.nwbQcam


parser = argparse.ArgumentParser(
    description='Convert QImaging .qcamraw epifluorescence data to NWB.')
parser.add_argument(
    'dataPath',
    help='Top-level directory containing per-subject data folders.')
parser.add_argument('--subjects', default='./data/animalList.csv',
                    metavar='PATH')
parser.add_argument('--experiments', default='./data/experimentMetadata.csv',
                    metavar='PATH')
parser.add_argument('--config', default='./configs/params_qcam.yaml',
                    metavar='PATH')
args = parser.parse_args()

print(f'dataPath:    {args.dataPath}')
print(f'subjects:    {args.subjects}')
print(f'experiments: {args.experiments}')
print(f'config:      {args.config}')

with open(args.config) as f:
    cfg = yaml.safe_load(f)

subjects = pd.read_csv(args.subjects).set_index('subject_id')
experiments = pd.read_csv(args.experiments).set_index('subject_id')
experiments['keywords'] = experiments['keywords'].apply(ast.literal_eval)

for experimentID, d in experiments.iterrows():
    print(f'processing: {experimentID}...')
    outputNWBpath = os.path.join(args.dataPath, experimentID,
                                 f'{experimentID}_qcam_DANDI.nwb')
    subject = lib.nwbScanImage.setSubject(
        subject_id=experimentID,
        age=f"P{subjects.loc[experimentID]['age']}D",
        species='Mus musculus',
        sex=subjects.loc[experimentID]['sex'],
        genotype=subjects.loc[experimentID]['genotype'],
        description=subjects.loc[experimentID]['description'],
    )
    lib.nwbQcam.genNWBfromQcamraw_pc(
        experimentID=experimentID,
        dataPath=args.dataPath,
        NWBoutputPath=outputNWBpath,
        subject=subject,
        session_description=d['session_description'],
        experiment_description=d['experiment_description'],
        keywords=d['keywords'],
        **cfg['nwb_file'],
        **cfg['imaging'],
        **cfg['qcam'],
    )