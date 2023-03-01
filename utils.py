#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Tuple, Union, Callable
from einops import rearrange

from ignite.metrics import Loss, Precision, Recall, RunningAverage, Accuracy, EpochMetric
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from torchaudio import transforms as audio_transforms
import torch_audiomentations as wavtransforms

# Some defaults for non-specified arguments in yaml
DEFAULT_ARGS = {
    'outputpath': 'experiments',
    'loss': 'BCELoss',
    'batch_size': 32,
    'sampler': 'balanced',
    'warmup_iters': 1000,
    'mixup': None,
    'num_workers': 2,  # Number of dataset loaders
    'spectransforms': {},  #Default no augmentation
    'wavtransforms': {},
    'early_stop': 10,  #Stop after M evaluations with low mAP
    'save':
    'best',  # if saving is done just every epoch, Otherwise any value is saving at test
    'epochs': 100,
    'n_saved': 4,
    'optimizer': 'Adam',
    'optimizer_args': {
        'lr': 0.001,
    },
}

def calculate_overall_lwlrap_sklearn(y_preds: torch.Tensor,
                                     y_targets: torch.Tensor) -> float:
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(y_true > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        y_true[nonzero_weight_sample_indices, :] > 0,
        y_pred[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


def compute_roc_auc(y_preds:torch.Tensor, y_targets:torch.Tensor) -> float:
    # print(y_preds.shape, y_targets.shape)
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    try:
        score = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        return 0.0
    return score


def compute_accuracy_with_noise(y_pred, y_tar):
    # Drop all "Noise" samples :(
    valid_idxs = torch.where(y_tar.max(-1).values != 0)
    y_true = y_tar[valid_idxs].argmax(-1).view(-1)
    y_pred = y_pred[valid_idxs].argmax(-1).view(-1)
    return accuracy_score(y_true, y_pred)

ALL_EVAL_METRICS = {
    'Accuracy':
    lambda: Accuracy(),
    # 'MultiClass_Accuracy':
    # lambda: Accuracy(output_transform=compute_accuracy_with_noise),
    'PositiveMultiClass_Accuracy':
    lambda: EpochMetric(compute_fn=compute_accuracy_with_noise),
    'Micro_Recall':
    lambda: EpochMetric(lambda y_pred, y_tar: recall_score(
        y_tar.numpy(), y_pred.numpy(), average='micro', zero_division=1),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Micro_Precision':
    lambda: EpochMetric(lambda y_pred, y_tar: precision_score(
        y_tar.numpy(), y_pred.numpy(), average='micro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Precision':
    lambda: EpochMetric(lambda y_pred, y_tar: precision_score(
        y_tar.numpy(), y_pred.numpy(), average=None, zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Macro_Precision':
    lambda: EpochMetric(lambda y_pred, y_tar: precision_score(
        y_tar.numpy(), y_pred.numpy(), average='macro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Macro_Recall':
    lambda: EpochMetric(lambda y_pred, y_tar: recall_score(
        y_tar.numpy(), y_pred.numpy(), average='macro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Recall':
    lambda: EpochMetric(lambda y_pred, y_tar: recall_score(
        y_tar.numpy(), y_pred.numpy(), average=None, zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Macro_F1':
    lambda: EpochMetric(lambda y_pred, y_tar: f1_score(
        y_tar.numpy(), y_pred.numpy(), average='macro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Micro_F1':
    lambda: EpochMetric(lambda y_pred, y_tar: f1_score(
        y_tar.numpy(), y_pred.numpy(), average='micro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'AUC':
    lambda: EpochMetric(compute_roc_auc, check_compute_fn=False),
    'BCELoss':
    lambda: Loss(torch.nn.BCELoss()),
    'CELoss':
    lambda: Loss(torch.nn.CrossEntropyLoss()),
    'mAP':
    lambda: EpochMetric(lambda y_pred, y_tar: np.nanmean(
        average_precision_score(y_tar.to('cpu').numpy(),
                                y_pred.to('cpu').numpy(),
                                average=None)),
                        check_compute_fn=False),
    'mAP_transform':
        lambda output_transform : EpochMetric(output_transform=output_transform,compute_fn=lambda y_pred, y_tar: np.nanmean(
        average_precision_score(y_tar.to('cpu').numpy(),
                                y_pred.to('cpu').numpy(),
                                average=None)),
                        check_compute_fn=False),

    'AP':
    lambda: EpochMetric(lambda y_pred, y_tar: average_precision_score(
        y_tar.to('cpu').numpy(), y_pred.to('cpu').numpy(), average=None),
                        check_compute_fn=False),
    'lwlwrap':
    lambda: EpochMetric(calculate_overall_lwlrap_sklearn,
                        check_compute_fn=False),
    # metrics.Lwlwrap(),
    'ErrorRate':
    lambda: EpochMetric(lambda y_pred, y_tar: 1. - np.nan_to_num(
        accuracy_score(y_tar.to('cpu').numpy(),
                       y_pred.to('cpu').numpy())),
                        check_compute_fn=False),
    # 'mAP@3':
    # metrics.mAPAt(k=3),
}


def metrics(metric_names: List[str]) -> Dict[str, EpochMetric]:
    '''
    Returns metrics given some metric names
    '''
    return {met: ALL_EVAL_METRICS[met]() for met in metric_names}

class DictWrapper(object):
    def __init__(self, adict):
        self.dict = adict

    def state_dict(self):
        return self.dict

    def load_state_dict(self, state):
        self.dict = state


def load_pretrained(model: torch.nn.Module, trained_model: dict):
    if 'model' in trained_model:
        trained_model = trained_model['model']
    model_dict = model.state_dict()
    # filter unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in trained_model.items() if (k in model_dict) and (
            model_dict[k].shape == trained_model[k].shape)
    }
    assert len(pretrained_dict) > 0, "Couldnt load pretrained model"
    # Found time positional embeddings ....
    if 'time_pos_embed' in trained_model.keys():
        pretrained_dict['time_pos_embed'] = trained_model['time_pos_embed']
        pretrained_dict['freq_pos_embed'] = trained_model['freq_pos_embed']

    logger.info(
        f"Loading {len(pretrained_dict)} Parameters for model {model.__class__.__name__}"
    )
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return model



def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = dict(yaml_config, **kwargs)
    # In case some arguments were not passed, replace with default ones
    for key, value in DEFAULT_ARGS.items():
        arguments.setdefault(key, value)
    return arguments


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def parse_wavtransforms(transforms_dict: Dict):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    transforms = []
    for trans_name, v in transforms_dict.items():
        transforms.append(getattr(wavtransforms, trans_name)(**v))

    return torch.nn.Sequential(*transforms)


def parse_spectransforms(transforms: Union[List, Dict]):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    if isinstance(transforms, dict):
        return torch.nn.Sequential(*[
            getattr(audio_transforms, trans_name)(**v)
            for trans_name, v in transforms.items()
        ])
    elif isinstance(transforms, list):
        return torch.nn.Sequential(*[
            getattr(audio_transforms, trans_name)(**v)
            for item in transforms
            for trans_name, v in item.items()
        ])
    else:
        raise ValueError("Transform unknown")



def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)

def mixup_single(x: torch.Tensor, lamb: torch.Tensor):
    """                                                                                     x: Tensor of shape ( batch_size, ... )         
    lamb: lambdas [0,1] of shape (batch_size)
    """

    x1 = rearrange(x.flip(0), 'b ... -> ... b')
    x2 = rearrange(x, 'b ... -> ... b')
    mixed = x1 * lamb + x2 * (1. - lamb)
    return rearrange(mixed, '... b -> b ...')


def mixup_lengths(x: torch.Tensor):
    """                                                                                     x: Tensor of shape ( batch_size, ... )         
    """

    return torch.maximum(x, x.flip(0))


def read_tsv_data(datafile: str, nrows: int = None, basename=True):
    df = pd.read_csv(datafile, sep='\s+', nrows=nrows).astype(str)
    assert 'hdf5path' in df.columns and 'filename' in df.columns and 'labels' in df.columns
    if any(df['labels'].str.contains(';')):
        df['labels'] = df['labels'].str.split(';').apply(
            lambda x: np.array(x, dtype=int))
    else:
        df['labels'] = df['labels'].apply(lambda x: [int(x)])
    if basename:
        # Just a hack to allow both GSC and audioset in one dataframe ....
        df['filename'] = df['filename'].apply(
            lambda x: x if 'Google_Speech_Commands' in x else Path(x).name)
    return df


def average_models(models: List[str]):
    model_res_state_dict = {}
    state_dict = {}
    has_new_structure = False
    for m in models:
        cur_state = torch.load(m, map_location='cpu')
        if 'model' in cur_state:
            has_new_structure = True
            model_params = cur_state.pop('model')
            # Append non "model" items, encoder, optimizer etc ...
            for k in cur_state:
                state_dict[k] = cur_state[k]
            # Accumulate statistics
            for k in model_params:
                if k in model_res_state_dict:
                    model_res_state_dict[k] += model_params[k]
                else:
                    model_res_state_dict[k] = model_params[k]
        else:
            for k in cur_state:
                if k in model_res_state_dict:
                    model_res_state_dict[k] += cur_state[k]
                else:
                    model_res_state_dict[k] = cur_state[k]

    # Average
    for k in model_res_state_dict:
        # If there are any parameters
        if model_res_state_dict[k].ndim > 0:
            model_res_state_dict[k] /= float(len(models))
    if has_new_structure:
        state_dict['model'] = model_res_state_dict
    else:
        state_dict = model_res_state_dict
    return state_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs="+")
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help="Output model (pytorch)")
    args = parser.parse_args()
    mdls = average_models(args.models)
    torch.save(mdls, args.output)
