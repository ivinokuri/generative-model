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
        """
        Init Dataloaders
        :param data_path: path to datafile
        :param batch_size: batch size of data loader
        :param window_size: window size for building labels for unlabeled data
        :param normalize: if need normalization
        :param features: range of features to use ex. np.r_[1:4]
        """
        super().__init__()
        self.normal_data_path = data_path
        self.anomaly_data_path = test_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.normalize = normalize

        self.train_val_data = pandas.read_csv(self.normal_data_path)
        self.test_data = pandas.read_csv(self.anomaly_data_path)
        if features is None or np.isscalar(features):
            self.n_features = len(self.train_val_data.columns)
            self.features_range = np.r_[0:len(self.train_val_data.columns)]
        else:
            self.n_features = len(features)
            self.features_range = features

        self.train = None
        self.test = None
        self.val = None
        self.predict = None
        self._has_setup_val = False
        self.scalers = {}

    def setup(self, stage=""):
        print(stage)
        # normalize
        if self.normalize:
            self.train_val_data = self.normalize_data(self.train_val_data)

        if stage == 'test':
            x_y_tupes = self._sliding_windows(self.test_data.values, 1)
            self.test = TopicsCountDataset(x_y_tupes)
        elif stage == 'val':
            x_y_tupes = self._sliding_windows(self.train_val_data.values, 1)
            self.val = TopicsCountDataset(x_y_tupes)
        else:
            # Create train and val tuples
            train_size = int(len(self.train_val_data.values) * 0.8)
            x_y_tupes = self._sliding_windows(self.train_val_data.values, self.window_size)
            self.train = TopicsCountDataset(x_y_tupes[:train_size])
            self.val = TopicsCountDataset(x_y_tupes[train_size:])


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=0)

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
    def _expanding_windows(data, window_size=10):
        x = []
        y = []
        for i in range(len(data) - window_size):
            _x = data[:(i + window_size)]
            _y = data[i + window_size]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    @staticmethod
    def _sliding_windows(data, window_size=10):
        x_y_tupes = []
        for i in range(len(data) - window_size):
            _x = data[i:(i + window_size)]
            _y = data[i + window_size]
            x_y_tupes.append((_x, _y))

        return x_y_tupes

class TopicCountPredictor(nn.Module):

    def __init__(self, features: int, hidden_size: int, num_layers: int):
        super(TopicCountPredictor, self).__init__()
        self.features = features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_layers=num_layers,
                            input_size=features,
                            hidden_size=hidden_size,
                            batch_first=True, dropout=0.2).to(device)
        self.linear = nn.Linear(in_features=hidden_size, out_features=features).to(device)


    def forward(self, input_data: torch.Tensor):
        self.lstm.flatten_parameters()
        out, (h_t, c_t) = self.lstm(input_data)
        pred = self.linear(h_t[1:])
        return pred, (h_t, c_t)

class TopicCountPredictorModule(pl.LightningModule):

    def __init__(self, features: int, hidden_size: int, num_layers: int, lr=0.001, loss="mse"):
        super().__init__()
        self.lr = lr
        self.loss = loss
        self.model = TopicCountPredictor(features, hidden_size, num_layers)
        if loss == 'mse':
            self.criteria = nn.MSELoss()
        elif loss == 'pois':
            self.criteria = nn.PoissonNLLLoss()

    def forward(self, x, y=None):
        output, (h, c) = self.model(x)
        loss = 0
        if y is not None:
            loss = self.criteria(output.squeeze(), y)
        return loss, output, (h, c)

    def training_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output, _ = self(sequence, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output, _= self(sequence, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def test_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output, _ = self(sequence, labels)
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
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    datamodule = TopicsCountDatamodule(
        data_path="../../../robot-data/new_data/normal/merged_normal_pick_count.csv",
        test_path="../../../robot-data/new_data/test/merged_pick_miss_cup_count.csv",
        batch_size=parsed_args.batch_size,
        window_size=parsed_args.window_size,
        normalize=parsed_args.norm)
    n_features = datamodule.n_features
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',
                                          filename='best-checkpoint',
                                          save_top_k=1,
                                          verbose=False,
                                          monitor='val_loss',
                                          mode='min')

    logger = TensorBoardLogger(save_dir='logs', name='topic-counts')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    loss_callback = LossAggregateCallback()
    trainer = Trainer(logger=logger,
                      enable_checkpointing=True,
                      callbacks=[checkpoint_callback, early_stopping_callback, loss_callback],
                      max_epochs=parsed_args.epochs,
                      gpus=torch.cuda.device_count())
    if parsed_args.train:

        model = TopicCountPredictorModule(features=n_features,
                                          hidden_size=(n_features if not parsed_args.hidden_states else parsed_args.hidden_states),
                                          num_layers=parsed_args.layers,
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
    trained_model = TopicCountPredictorModule.load_from_checkpoint(last_model,
                                                                   features=n_features,
                                                                   hidden_size=(n_features if not parsed_args.hidden_states else parsed_args.hidden_states),
                                                                   num_layers=parsed_args.layers)
    trained_model.freeze()
    datamodule.setup('val')
    normal_set = datamodule.val
    normal_losses = []
    for item in tqdm(normal_set):
        sequence = item['sequence']
        label = item['label']
        loss, output, _ = trained_model(sequence.unsqueeze(dim=0), label)
        normal_losses.append(loss.item())

    datamodule.setup('test')
    anomaly_set = datamodule.test
    anomaly_losses = []
    for item in tqdm(anomaly_set):
        sequence = item['sequence']
        label = item['label']
        loss, output, _ = trained_model(sequence.unsqueeze(dim=0), label)
        anomaly_losses.append(loss. item())

    fig, axs = plt.subplots(1, figsize=(15, 10))
    axs.plot(range(len(normal_losses)), normal_losses, color="g", label="Normal Loss")
    axs.plot(range(len(anomaly_losses)), anomaly_losses, color="r", alpha=0.7, label="Anomaly Loss")
    axs.legend()
    plt.legend()
    plt.savefig('plots/anomaly_loss_' + date_time + '.png')
    plt.show()
    plt.close()

    fig, axs = plt.subplots(1, figsize=(17, 10))
    n, bins, patches = axs.hist(normal_losses, 300, facecolor='g', alpha=0.5, density=True, stacked=True,
                                label="Normal loss$")
    n, bins, patches = axs.hist(anomaly_losses, 300, facecolor='r', alpha=0.5, density=True, stacked=True,
                                label="Anomaly loss$")
    axs.legend()
    plt.legend()
    plt.savefig('plots/hist_loss_' + date_time + '.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--train", type=bool, default=True, help="Retrain model pass True")
    parser.add_argument("--norm", type=bool, default=False, help="Normalizing data")
    parser.add_argument("--loss", type=str, default='mse', help="Loss function (mse, pois)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--window_size", type=int, default=10, help="window size")
    parser.add_argument("--hidden_states", type=int, default=None, help="Number of hidden states")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")

    parsed_args, _ = parser.parse_known_args()

    main(parsed_args)