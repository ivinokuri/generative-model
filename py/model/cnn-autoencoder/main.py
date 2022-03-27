"""
"""
import argparse
import csv
import logging
import os
import sys
from typing import AnyStr, List

import numpy as np
import torch
from torchinfo import summary

import model

VALIDATION_FRACTION = 1.0  # of remaining data
TEST_FRACTION = 0.05
WINDOWS_SIZE = 10
BATCH_SIZE = 128
NUM_EPOCHS = 100
BURNIN = 5
LEARNING_RATE = 0.001
PATIENCE = 5
EPSILON = 1E-8  # Adam's stabilizer
SEED = 0

logging.basicConfig(
    format='%(filename)s.%(lineno)s %(funcName)s: %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(logger)


def get_args(args=None):
    """Parse command line arguments. """

    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="robot anomaly detection with cnn")

    parser.add_argument("-w",
                        "--windows-size",
                        type=int,
                        default=WINDOWS_SIZE,
                        help=f"sample len, {WINDOWS_SIZE} by default")

    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=BATCH_SIZE,
                        help=f"batch size, {BATCH_SIZE} by default")
    parser.add_argument("-r",
                        "--learning-rate",
                        type=float,
                        default=LEARNING_RATE,
                        help=f"learning rate, {LEARNING_RATE} by default")
    parser.add_argument("-e",
                        "--epsilon",
                        type=float,
                        default=EPSILON,
                        help=f"Adam's epsilon, {EPSILON} by default")
    parser.add_argument("-t",
                        "--test-fraction",
                        type=float,
                        default=TEST_FRACTION,
                        help="fractions of data for validation "
                        f"and testing, {TEST_FRACTION} by default")
    parser.add_argument("-v",
                        "--validation-fraction",
                        type=float,
                        default=VALIDATION_FRACTION,
                        help="fraction of data for validation, "
                        f"{VALIDATION_FRACTION} of remaining data by default")
    parser.add_argument("-n",
                        "--num-epochs",
                        type=int,
                        default=NUM_EPOCHS,
                        help="number of training epochs, "
                        f"{NUM_EPOCHS} by default")
    parser.add_argument("-u",
                        "--burnin",
                        type=int,
                        default=BURNIN,
                        help="minimum epochs before the model is saved, "
                        f"{BURNIN} by default")
    parser.add_argument("-p",
                        "--patience",
                        type=int,
                        default=PATIENCE,
                        help="number of epochs without progress "
                        f"for early stopping, {PATIENCE} by default")
    parser.add_argument('--no-cuda',
                        default=False,
                        action="store_true",
                        help='disables CUDA')

    parser.add_argument("fdata", help="path to data csv folder")
    parser.add_argument("fanomaly", help="path to anomaly data csv folder")

    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda") if args.cuda else torch.device("cpu")
    logger.info(f"args.device: {args.device}")

    return args


def vectorization(array: np.ndarray, windows_size: int) -> np.ndarray:
    swz = np.expand_dims(np.arange(windows_size), 0)
    amt = np.expand_dims(np.arange(len(array) - windows_size + 1), 0).T
    return array[swz + amt]


def make_data(paths: List, windows_size: int) -> torch.Tensor:
    dataset = []
    for path in paths:
        with open(path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        idx = [i for i, col in enumerate(header) if 'counter' in col.lower()]
        array = np.array(rows, dtype=np.float64)
        array = array[:, idx]
        dataset.append(vectorization(array, windows_size))
    data = np.concatenate(dataset)
    x, y, z = data.shape
    # data = data.reshape(x, 1, y, z)

    return torch.from_numpy(data).float()


def split_data(data, args):
    """Splits data in into train, validation, and tes sets.
    Returns the train set.
    """
    logger.info(f'{data.size}')
    ntest = round(data.shape[0] * args.test_fraction)
    nvald = ntest
    leftover = (data.shape[0] - ntest - nvald) % args.batch_size
    # training may misbehave if leftover << batch_size;
    # move the leftover to the validation data
    nvald += leftover
    train_data = data[ntest:-nvald]
    validation_data = data[-nvald:]
    test_data = data[:ntest]
    return train_data, validation_data, test_data


def get_paths(folder: AnyStr) -> List[AnyStr]:
    return [os.path.join(folder, file) for file in os.listdir(folder)]


def main():

    args = get_args()

    pdata = get_paths(args.fdata)
    data = make_data(pdata, args.windows_size)
    panomaly = get_paths(args.fanomaly)
    anomaly = make_data(panomaly, args.windows_size)

    # Split into train, validation, and test
    train_data, validation_data, test_data = split_data(data, args)
    # Create data loaders
    logger.info(f'len train data {len(train_data)}')
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    logger.info(f'len train loader {len(train_loader)}')
    validation_loader = torch.utils.data.DataLoader(validation_data,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    anomaly_loader = torch.utils.data.DataLoader(anomaly,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Set the random seed for reproducible results
    torch.manual_seed(SEED)

    ### Initialize the two networks
    d = 50

    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    logger.info(f"{args.windows_size}, {data.shape}")

    encoder = model.Encoder(
        encoded_space_dim=d,
        sqln=args.windows_size,
    )
    decoder = model.Decoder(
        encoded_space_dim=d,
        sqln=args.windows_size,
    )

    params_to_optimize = [{
        'params': encoder.parameters()
    }, {
        'params': decoder.parameters()
    }]
    optim = torch.optim.Adam(params_to_optimize,
                             lr=args.learning_rate,
                             weight_decay=1e-05)
    # Move both the encoder and the decoder to the selected device
    encoder.to(args.device)
    decoder.to(args.device)

    diz_loss = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'anomaly_loss': []
    }
    for epoch in range(args.num_epochs):
        train_loss = model.train_epoch(encoder, decoder, args.device,
                                       train_loader, loss_fn, optim)
        val_loss = model.test_epoch(encoder, decoder, args.device,
                                    validation_loader, loss_fn)
        test_loss = model.test_epoch(encoder, decoder, args.device, test_loader,
                                     loss_fn)
        anomaly_loss = model.test_epoch(encoder, decoder, args.device,
                                        anomaly_loader, loss_fn)
        logger.info(f'\n EPOCH {epoch + 1}/{args.num_epochs} '
                    f'\t train loss {train_loss} '
                    f'\t val loss {val_loss}'
                    f'\t test loss {test_loss}'
                    f'\t anomaly loss {anomaly_loss}')
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        diz_loss['test_loss'].append(test_loss)
        diz_loss['anomaly_loss'].append(anomaly_loss)

    logger.info(diz_loss)


if __name__ == '__main__':
    main()
