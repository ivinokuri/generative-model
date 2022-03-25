#!/usr/bin/env python3

import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
from autoencoder import Autoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

N_EPOCHS = 100
BATCH_SIZE=20
HIDDEN_SIZE=79

date_time=""

NORMAL_DATA_BASE_PATH = "../../../robot-data/new_data/normal/"
NORMAL_DATA = ['building/counts_only', 'cans/counts_only', 'corr/counts_only', 'pick/counts_only']

ANOMALY_DATA_BASE_PATH = "../../../robot-data/new_data/test/"
ANOMALY_DATA = ['laser_fault/build/counts_only', 'laser_fault/cans/counts_only', 'laser_fault/corr/counts_only',
                'obs/cans/counts_only', 'obs/corr/counts_only', 'obs/cans/counts_only',
                'pick/miss_cup/counts_only', 'pick/restricted_vision/counts_only', 'pick/stolen/counts_only',
                'pick/stuck/counts_only', 'software_fault/counts_only', 'velocity_attack/counts_only']

class TopicsCountDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=Tensor(sequence),
            label=Tensor(label)
        )
    
class TopicsCountDatamodule(pl.LightningDataModule):

    def __init__(self, data_dir_path: str = "", test_dir_path="", batch_size=10,
                 window_size=10,
                 normalize=True):
        super(TopicsCountDatamodule, self).__init__()
        self.scalers = {}

        self.normal_data_path = data_dir_path
        self.anomaly_data_path = test_dir_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.normalize = normalize
        self.n_features = 0

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
        if self.normal_data_path:
            data_files = list_files(self.normal_data_path)
            for f in data_files:
                file_data = pandas.read_csv(f)
                self.n_features = len(file_data.columns)
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
        if self.anomaly_data_path:
            data_files = list_files(self.anomaly_data_path)
            for f in data_files:
                file_data = pandas.read_csv(f)
                if self.normalize:
                    _, tail = os.path.split(f)
                    file_data = self.normalize_data(file_data, name=tail)
                x_y_tuples = self._sliding_windows(file_data.values, 1)
                self.test_data.append(x_y_tuples)

    def setup(self, stage='') -> None:
        if stage == 'fit':
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
        return DataLoader(ConcatDataset(self.train), batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(ConcatDataset(self.test), batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(ConcatDataset(self.val), batch_size=self.batch_size, shuffle=False)

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
        for i in range(len(data) - window_size - 1):
            _x = data[i:(i + window_size)]
            _y = data[i+1: i + window_size + 1]
            x_y_tupes.append((_x, _y))

        return x_y_tupes

class TopicCountAutoencoderModule(pl.LightningModule):

    def __init__(self, features: int, lr: float=0.001, loss: str="mse"):
        super().__init__()
        self.lr = lr
        self.loss = loss
        self.model = Autoencoder(features, int(features/4), features)
        if loss == 'mse':
            self.criteria = nn.MSELoss()

    def forward(self, x, y=None):
        output = self.model(x)
        loss = 0
        if y is not None:
            loss = self.criteria(output, y)
        return loss, output

    def training_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequence, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequence, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def test_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequence, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

    def get_criteria(self):
        return self.criteria


class LossAggregateCallback(pl.Callback):

    def __init__(self):
        self.train_losses = []
        self.validation_losses = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_losses.append(trainer.logged_metrics['train_loss_epoch'])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.validation_losses.append(trainer.logged_metrics['val_loss'])

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        global date_time
        fig, axs = plt.subplots(1)
        axs.plot(range(len(self.train_losses)), self.train_losses, label="Train loss")
        axs.plot(range(len(self.validation_losses)), self.validation_losses, label="Validation loss")
        axs.set_title('Loss')
        axs.legend()
        plt.legend()
        if not os.path.isdir('./plots'):
            os.mkdir('./plots')
        plt.savefig('plots/loss_' + date_time + '.png')
        plt.show()
        plt.close()


def main(parsed_args):
    now = datetime.now()
    global date_time
    normal_dir_paths = []
    anomaly_dir_paths = []
    for nd in NORMAL_DATA:
        normal_dir_paths.append({nd.replace('/', '_'): NORMAL_DATA_BASE_PATH + nd})

    for ad in ANOMALY_DATA:
        anomaly_dir_paths.append({ad.replace('/', '_'): ANOMALY_DATA_BASE_PATH + ad})
    print(normal_dir_paths)
    print(anomaly_dir_paths)
    # heat map for all scenarios
    # mean and std
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    datamodule = TopicsCountDatamodule(
        data_dir_path="../../../robot-data/new_data/normal/pick/counts_only/",
        test_dir_path="../../../robot-data/new_data/test/pick/miss_cup_counts/",
        batch_size=parsed_args.batch_size,
        window_size=parsed_args.window_size,
        normalize=parsed_args.norm)
    datamodule.prepare_data()
    n_features = datamodule.n_features
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',
                                          filename='best-checkpoint',
                                          save_top_k=1,
                                          verbose=False,
                                          monitor='val_loss',
                                          mode='min')

    logger = TensorBoardLogger(save_dir='logs', name='topic-counts')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)
    loss_callback = LossAggregateCallback()
    trainer = Trainer(logger=logger,
                      enable_checkpointing=True,
                      callbacks=[checkpoint_callback, early_stopping_callback, loss_callback],
                      max_epochs=parsed_args.epochs,
                      gpus=torch.cuda.device_count())
    if parsed_args.train:
        model = TopicCountAutoencoderModule(features=n_features,
                                          lr=parsed_args.lr,
                                          loss=parsed_args.loss)

        trainer.fit(model, datamodule=datamodule)

    list_of_files = filter(os.path.isfile,
                           glob.glob('checkpoints/' + '*'))
    last_model = sorted(list_of_files,
                        key=os.path.getmtime)
    print(last_model)
    last_model = last_model[len(last_model) - 1]
    print(last_model)
    trained_model = TopicCountAutoencoderModule.load_from_checkpoint(last_model,
                                                                   features=n_features)

    trained_model.freeze()
    datamodule.setup('val')
    normal_set = datamodule.val
    normal_losses = []
    for d in normal_set:
        nl = []
        for item in tqdm(d):
            sequence = item['sequence']
            label = item['label']
            loss, output = trained_model(sequence.unsqueeze(dim=0), label.unsqueeze(dim=0))
            nl.append(loss.item())
        normal_losses.append(nl)

    datamodule.setup('test')
    anomaly_set = datamodule.test
    anomaly_losses = []
    for d in anomaly_set:
        nl = []
        for item in tqdm(d):
            sequence = item['sequence']
            label = item['label']
            loss, output = trained_model(sequence.unsqueeze(dim=0), label.unsqueeze(dim=0))
            nl.append(loss.item())
        anomaly_losses.append(nl)

    i = 0
    for nl in normal_losses:
        fig, axs = plt.subplots(1, figsize=(15, 10))
        axs.plot(range(len(nl)), nl, color="g", label="Normal Loss")
        axs.legend()
        plt.legend()
        plt.savefig('plots/normal_loss_' + date_time + '_' + str(i) + '.png')
        plt.show()
        plt.close()
        i += 1

    i = 0
    for al in anomaly_losses:
        fig, axs = plt.subplots(1, figsize=(15, 10))
        axs.plot(range(len(al)), al, color="r", label="Anomaly Loss")
        axs.legend()
        plt.legend()
        plt.savefig('plots/anomaly_loss_' + date_time + '_' + str(i) + '.png')
        plt.show()
        plt.close()
        i += 1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--train", type=bool, default=False, help="Retrain model pass True")
    parser.add_argument("--norm", type=bool, default=True, help="Normalizing data")
    parser.add_argument("--loss", type=str, default='mse', help="Loss function (mse, pois)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument("--window_size", type=int, default=10, help="window size")

    parsed_args, _ = parser.parse_known_args()

    main(parsed_args)