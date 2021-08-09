import torch

from model.autoencoder.autoencoder import Autoencoder
import pandas
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from tqdm import tqdm
import os
print(os.getcwd())
# os.makedirs(os.getcwd() + '\\res')

scalers = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def normalize_data(original_data):
    norm_data = original_data
    for i in original_data.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        s_s = scaler.fit_transform(norm_data[i].values.reshape(-1, 1))
        s_s = np.reshape(s_s, len(s_s))
        scalers['scaler_' + i] = scaler
        norm_data[i] = s_s
    return norm_data

def unnormalize_data(original_data, reconstructed_data):
    restored_data = []
    for (i, name) in enumerate(original_data.columns):
        restored_data.append(np.array(scalers['scaler_' + name].inverse_transform(reconstructed_data[:,i].reshape(-1, 1))).flatten())
    return np.array(restored_data)


def split_data(data, window):
    """
    Data split for train and labels with sliding window
    :param data: original data
    :param window: subsequence length
    :return:
    """
    train_x = data[:window + 1]
    x, y = sliding_windows(train_x, 1)
    train_x = Variable(torch.Tensor(np.array(x)))
    train_y = Variable(torch.Tensor(np.array(y)))
    return train_x, train_y


def sliding_windows(data, seq_length):
    """
    Sliding window transformation for the data
    :param data: original data
    :param seq_length: window size
    :return: two arrays of x and y
    """
    x = []
    y = []
    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

def train(model, train_data, test_data):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    objective = nn.MSELoss().to(device)
    history = []
    test_losses = []
    for e in tqdm(range(100)):
        model.train()
        loss = 0
        test_loss = 0
        for i in tqdm(range(len(train_data))):
            x = train_data[i]
            x = torch.Tensor(np.array([x])).to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = objective(reconstructed, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if i % 1000 == 0:
                print("\n========== \nEpoch: ", e, " Seq: ", i, "\n=============")
        history.append(loss.item())
        with torch.no_grad():
            model.eval()
            x = torch.Tensor(np.array([test_data])).to(device)
            reconstruction = model(x)
            test_loss = objective(reconstruction, x)
            test_losses.append(test_loss.item())
        # if e % 10 == 0:
        print("================================")
        print('Epoch: ', e);
        print("Loss: ", loss.item())
        print("Test Loss: ", test_loss.item())
        print("================================")
    return history, test_losses

def predict(model, dataset):
  predictions = []
  with torch.no_grad():
    model = model.eval()
    x = torch.Tensor(np.array([dataset])).to(device)
    reconstruction = model(x)
  return reconstruction.squeeze(0)

def main():
    DATA = pandas.read_csv('../../../rnn/merged.csv')
    n_features = len(DATA.columns)
    print(n_features)
    norm_data = normalize_data(DATA.copy()).values
    print(norm_data[0])
    autoencoder = Autoencoder(2, n_features, 35, n_features, 60, n_features).to(device)
    train_size = np.int(len(norm_data) * 0.8)
    train_data, test_data = norm_data[:train_size], norm_data[train_size:]
    train_x, train_y = sliding_windows(train_data, 60)
    h, l = train(autoencoder, train_x, test_data)

    print(h)
    print(l)
    fig, axs = plt.subplots(1)

    axs.plot(range(len(h)), h, label="Train loss")
    axs.plot(range(len(l)), l, label="Test loss")
    axs.set_title('Losses')
    axs.legend()
    plt.legend()
    plt.show()
    torch.save(autoencoder.state_dict(), "my_model.model")
    reconstruction = predict(autoencoder, test_data)
    reconstruction = unnormalize_data(DATA, reconstruction.cpu().numpy())

    original = DATA.values[:len(test_data)]
    for i in range(len(reconstruction)):
        plt.figure(figsize=(20, 20))
        plt.plot(range(len(reconstruction[i])), reconstruction[i], label="Reconstructed " + DATA.columns[i])
        plt.plot(range(len(original[:,i])), original[:,i], label="Original " + DATA.columns[i])
        plt.legend()
        plt.savefig(os.getcwd() + '\\res\\' + str(i))
        plt.close()

if __name__ == "__main__":
    main()