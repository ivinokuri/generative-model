#!/usr/bin/env python3
import torch
from torch.autograd import Variable
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
