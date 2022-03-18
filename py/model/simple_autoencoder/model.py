#!/usr/bin/env python3

import torch
from torch import nn, Tensor, optim
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas
import os
import glob
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

N_EPOCHS = 60
BATCH_SIZE=20
HIDDEN_SIZE=79

date_time=""

class TopicsCountDataset(IterableDataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        pass

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=Tensor(sequence),
            label=Tensor(np.array(label))
        )
    
class TopicsCountDatamodule(pl.LightningDataModule):

    def __init__(self, data_path: str = "", test_path="", batch_size=10,
                 window_size=10,
                 normalize=True,
                 features=None):
        super(TopicsCountDatamodule, self).__init__()
        self.scalers = {}

    def setup(self, stage='') -> None:
        pass

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def normalize_data(self, data):
        norm_data = data
        for i in data.columns:
            scaler = None
            if ('scaler_' + i) not in self.scalers:
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                scaler = self.scalers['scaler_' + i]
            s_s = scaler.fit_transform(norm_data[i].values.reshape(-1, 1))
            s_s = np.reshape(s_s, len(s_s))
            self.scalers['scaler_' + i] = scaler
            norm_data[i] = s_s
        return norm_data

    def unnormalize_data(self, columns, reconstructed_data):
        restored_data = []
        for (i, name) in enumerate(columns):
            restored_data.append(np.array(
                self.scalers['scaler_' + name].inverse_transform(reconstructed_data[:, i].reshape(-1, 1))).flatten())

        return np.array(restored_data)

    @staticmethod
    def _sliding_windows(data, window_size=10):
        x_y_tupes = []
        for i in range(len(data) - window_size):
            _x = data[i:(i + window_size)]
            _y = data[i + window_size]
            x_y_tupes.append((_x, _y))

        return x_y_tupes