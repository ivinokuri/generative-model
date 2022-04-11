#!/usr/bin/env python3

import torch
from torch.autograd import Variable

from os import listdir
from os.path import isfile, join
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORMAL_DATA_BASE_PATH = "../../../robot-data/new_data/normal/"
NORMAL_DATA = ['building/counts_only/', 'cans/counts_only/', 'corr/counts_only/', 'pick/counts_only/']

ANOMALY_DATA_BASE_PATH = "../../../robot-data/new_data/test/"
ANOMALY_DATA = [['laser_fault/build/counts_only/'],
                ['laser_fault/cans/counts_only/', 'obs/cans/counts_only/', 'obs/cans/counts_only/'],
                ['laser_fault/corr/counts_only/', 'obs/corr/counts_only/'],
                ['pick/miss_cup/counts_only/', 'pick/restricted_vision/counts_only/',
                 'pick/stolen/counts_only/', 'pick/stuck/counts_only/']]


def sort_predicate(value):
    nums = re.findall(r'\d+', value)
    return int(nums[0])

def list_files(dir, sorted=True):
    onlyfiles = [dir + f for f in listdir(dir) if isfile(join(dir, f))]
    if sorted:
        onlyfiles.sort(key=sort_predicate)
    return onlyfiles


def init_hidden(x: torch.Tensor, hidden_size: int, num_layers: int = 1):
    """
    Initialize hidden.
    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    return Variable(torch.zeros(num_layers, x.size(0), hidden_size)).to(device)

def get_file_paths():
    normal_dir_paths = []
    i = 0
    for nd in NORMAL_DATA:
        anomaly_dir_paths = []
        for ad in ANOMALY_DATA[i]:
            anomaly_dir_paths.append({ad.replace('/', '_'): ANOMALY_DATA_BASE_PATH + ad})
        normal_dir_paths.append({nd.replace('/', '_'): [NORMAL_DATA_BASE_PATH + nd, anomaly_dir_paths]})
        i += 1

    return normal_dir_paths