#!/usr/bin/env python3

import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas

scalers = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class TopicsCountDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=Tensor(sequence.to_numpy()),
            label=Tensor(label.float())
        )

class TopicsCountDatamodel(pl.LightningDataModule):

    def __init__(self, normal_data_path: str = "", anormaly_data_path: str = ""):
        super().__init__()
        self.normal_data_path = normal_data_path
        self.anomaly_data_path = anormaly_data_path
        self.train_test_data = []
        self.val_data = []

        self.train = None
        self.test = None
        self.val = None

        self.batch_size = 0
        self.scalers = {}
        self.prepare_data()

    def setup(self, batch_size=10, window_size=10, normalize=True):
        # load
        self.train_test_data = pandas.read_csv(self.normal_data_path)
        self.val_data = pandas.read_csv(self.anomaly_data_path)
        self.batch_size = batch_size
        # normalize
        if normalize:
            self.train_test_data = self.normalize_data(self.train_test_data)
            self.val_data = self.normalize_data(self.val_data)

        # Create test train tuples
        train_size = int(len(self.train_test_data.values) * 0.8)
        x, y = self._sliding_windows(self.train_test_data.values, window_size)
        self.train = TopicsCountDataset(zip(x[:train_size], y[:train_size]))
        self.test = TopicsCountDataset(zip(x[train_size], y[train_size]))

        # Create val tuples
        x, y = self._sliding_windows(self.val_data.values, window_size)
        self.val = TopicsCountDataset(zip(x, y))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=2)

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
        x = []
        y = []
        for i in range(len(data) - window_size):
            _x = data[i:(i + window_size)]
            _y = data[i + window_size]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

class TopicCountPredictor(nn.Module):

    def __init__(self, features: int, hidden_size: int, num_layers: int):
        """
        Initialize the model.
        Args:
            config:
            input_size: (int): size of the input
        """
        super(TopicCountPredictor, self).__init__()
        self.features = features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_layers=num_layers, input_size=features,
                            hidden_size=hidden_size, batch_first=True, dropout=0.2).to(device)


    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.
        Args:
            input_data: (torch.Tensor): tensor of input daa
        """

        out, (h_t, c_t) = self.lstm(input_data)
        return out, (h_t, c_t)


class TopicCountPredictorModule(pl.LightningDataModule):

    def __init__(self, features: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.model = TopicCountPredictor(features, hidden_size, num_layers)
        self.criteria = nn.MSELoss()

    def forward(self, x, y=None):
        output = self.model(x)
        loss = 0
        if y is not None:
            loss = self.criteria(output, y.unsqueeze(dim=1))
        return loss, output

    def training_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequence, labels)
        self.log("Training loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequence, labels)
        self.log("Validation loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def test_step(self, batch, index):
        sequence = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequence, labels)
        self.log("Test loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.0001)

def main():
    datamodel = TopicsCountDatamodel(normal_data_path="../../../robot-data/new_data/normal/merged_normal_pick_count.csv",
                                     anormaly_data_path="../../../robot-data/new_data/test/merged_pick_miss_cup_count.csv")
    datamodel.setup()
    print(datamodel.train_dataloader())



if __name__ == '__main__':
    main()