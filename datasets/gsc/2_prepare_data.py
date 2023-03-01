#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import argparse

# Don't like this hacky stuff
import sys
path_root = Path(__file__).parent.parent
sys.path.append(str(path_root))

from utils.dump_audio_to_hdf5 import dump_waves

LABEL_MAPS_GSC_AUDIOSET = {
    'bed': 0,
    'bird': 0,
    'cat': 0,
    'dog': 0,
    'down': 527,
    'eight': 0,
    'five': 0,
    'four': 0,
    'go': 528,
    'happy': 0,
    'house': 0,
    'left': 529,
    'marvin': 0,
    'nine': 0,
    'no': 530,
    'off': 531,
    'on': 532,
    'one': 0,
    'right': 533,
    'seven': 0,
    'sheila': 0,
    'six': 0,
    'stop': 534,
    'three': 0,
    'tree': 0,
    'two': 0,
    'up': 535,
    'wow': 0,
    'yes': 536,
    'zero': 0
}

parser = argparse.ArgumentParser()
parser.add_argument('gsc_root_path',
                    type=Path,
                    default='gsc_raw_data',
                    nargs='?')
parser.add_argument('output_root_dir', type=Path, default='data', nargs='?')
args = parser.parse_args()

valid_df = pd.read_csv(args.gsc_root_path / 'validation_list.txt',
                       sep=' ',
                       names=['fn'])
test_df = pd.read_csv(args.gsc_root_path / 'testing_list.txt',
                      sep=' ',
                      names=['fn'])

data_store = []
for f in Path(args.gsc_root_path).glob('**/*wav'):
    #Structure is:
    #data/
    #----/label/
    #----/-----/a.wav
    label_name = f.parent.name
    label = LABEL_MAPS_GSC_AUDIOSET.get(label_name)
    # Path that is identical to validation_df and test_df
    fn = f"{f.parent.name}/{f.name}"
    # if label is not None and label != 0:
    # label = f"0;{label}"
    data_store.append({
        'filename': str(f.absolute()),
        'labels': label,
        'fn': fn
    })
df = pd.DataFrame(data_store).dropna(axis='rows')
# Due to NAn, the values are in float here
df['labels'] = df['labels'].astype(int)

# Merge with test/valid splits
test_df = pd.merge(df, test_df, on='fn').drop('fn', axis=1)
valid_df = pd.merge(df, valid_df, on='fn').drop('fn', axis=1)
valid_test_fn = pd.concat((valid_df, test_df))['filename'].values

train_df = df[~df['filename'].isin(valid_test_fn)].drop('fn', axis=1).copy()
train_df.loc[train_df['labels'] != 0,
             'labels'] = "0;" + train_df.loc[train_df['labels'] != 0,
                                             'labels'].astype(str)

all_data_dfs = {'train': train_df, 'valid': valid_df, 'test': test_df}

args.output_root_dir.mkdir(exist_ok=True, parents=True)

output_labels = args.output_root_dir / 'labels'
output_hdf5 = args.output_root_dir / 'hdf5'

output_labels.mkdir(exist_ok=True, parents=True)
output_hdf5.mkdir(exist_ok=True, parents=True)

for dataname, df in all_data_dfs.items():
    output_h5_path = output_hdf5 / f'{dataname}.h5'
    df['hdf5path'] = output_hdf5.absolute()
    print(
        f"Dumping wav to hdf5 for {dataname} [len {len(df)}] to {output_h5_path}"
    )
    dump_waves(df, output_h5_path, use_fullname=True)
    df.to_csv(output_labels / f'{dataname}_gsc_aslabels.tsv',
              sep='\t',
              index=False)
