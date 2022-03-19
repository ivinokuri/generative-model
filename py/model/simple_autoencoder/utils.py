#!/usr/bin/env python3

import torch
from torch.autograd import Variable

from os import listdir
from os.path import isfile, join
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
