#!/usr/bin/env python3

import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader
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
from utils import list_files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

N_EPOCHS = 60
BATCH_SIZE=20
HIDDEN_SIZE=79

date_time=""

class TopicsCountDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    # def __iter__(self):
    #     pass

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=Tensor(sequence),
            label=Tensor(np.array(label))
        )
    
class TopicsCountDatamodule(pl.LightningDataModule):

    def __init__(self, data_dir_path: str = "", test_dir_path="", batch_size=10,
                 window_size=10,
                 normalize=True,
                 features=None):
        super(TopicsCountDatamodule, self).__init__()
        self.scalers = {}

        self.normal_data_path = data_dir_path
        self.anomaly_data_path = test_dir_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.normalize = normalize

        self.train_val_data = []
        self.validation_data = []
        self.test_data = []

        self.train = []
        self.test = []
        self.val = []
        self.predict = None
        self._has_setup_train = False
        self._has_setup_val = False
        self._has_setup_test = False

    def prepare_data(self) -> None:
        # load train and validation data
        data_files = list_files(self.normal_data_path)
        for f in data_files:
            file_data = pandas.read_csv(f)
            if self.normalize:
                _, tail = os.path.split(f)
                file_data = self.normalize_data(file_data, name=tail)
            train_size = int(len(file_data.values) * 0.8)
            x_y_tuples = self._sliding_windows(file_data.values, self.window_size)
            self.train_val_data.append((x_y_tuples[:train_size], x_y_tuples[train_size:]))
            # full set for calculating loss at the end of the training
            x_y_tuples = self._sliding_windows(file_data.values, 1)
            self.validation_data.append(x_y_tuples)

        # load test data
        data_files = list_files(self.anomaly_data_path)
        for f in data_files:
            file_data = pandas.read_csv(f)
            if self.normalize:
                _, tail = os.path.split(f)
                file_data = self.normalize_data(file_data, name=tail)
            x_y_tuples = self._sliding_windows(file_data.values, 1)
            self.test_data.append(x_y_tuples)

    def setup(self, stage='') -> None:
        if stage == 'train':
            for (t, v) in self.train_val_data:
                self.train.append(TopicsCountDataset(t))
                self.val.append(TopicsCountDataset(v))
        elif stage == 'test':
            for t in self.test_data:
                self.test.append(TopicsCountDataset(t))
        elif stage == 'val':
            self.val = []
            for t in self.validation_data:
                self.val.append(TopicsCountDataset(t))

    def train_dataloader(self):
        return [DataLoader(d, batch_size=self.batch_size, shuffle=False)
                for d in self.train]

    def test_dataloader(self):
        return [DataLoader(d, batch_size=self.batch_size, shuffle=False)
                for d in self.test]

    def val_dataloader(self):
        return [DataLoader(d, batch_size=self.batch_size, shuffle=False)
                for d in self.val]

    def normalize_data(self, data, name=""):
        norm_data = data
        for i in data.columns:
            scaler = None
            if (name + '_scaler_' + i) not in self.scalers:
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                scaler = self.scalers[name + '_scaler_' + i]
            s_s = scaler.fit_transform(norm_data[i].values.reshape(-1, 1))
            s_s = np.reshape(s_s, len(s_s))
            self.scalers[name + '_scaler_' + i] = scaler
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

def main(parsed_args):
    datamodule = TopicsCountDatamodule(
        data_dir_path="../../../robot-data/new_data/normal/pick/counts_only/",
        test_dir_path="../../../robot-data/new_data/test/pick/miss_cup_counts/",
        batch_size=parsed_args.batch_size,
        window_size=parsed_args.window_size,
        normalize=parsed_args.norm)
    datamodule.prepare_data()
    datamodule.setup('train')
    trainLoader = datamodule.train_dataloader()
    for batch in trainLoader:
        for b in batch:
            print(b)

    valLoader = datamodule.val_dataloader()
    datamodule.setup('test')
    testLoader = datamodule.test_dataloader()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--train", type=bool, default=True, help="Retrain model pass True")
    parser.add_argument("--norm", type=bool, default=True, help="Normalizing data")
    parser.add_argument("--loss", type=str, default='mse', help="Loss function (mse, pois)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--window_size", type=int, default=10, help="window size")
    parser.add_argument("--hidden_states", type=int, default=158, help="Number of hidden states")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")

    parsed_args, _ = parser.parse_known_args()

    main(parsed_args)