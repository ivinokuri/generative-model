"""
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

logger = logging.getLogger('__main__')


def shape2d(shape: Tuple,
            kernel_size: _size_2_t,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            stride: _size_2_t = 1):

    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)

    h = int((shape[0] + 2 * padding[0] - dilation[0] *
             (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = int((shape[1] + 2 * padding[1] - dilation[1] *
             (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w


def shape1d(
        L: int,  # pylint: disable=invalid-name
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        stride: int = 1):

    h = int((L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return h


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, features: int):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(79, 8, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv1d(8, 16, 3, stride=2, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )

        # pylint: disable=invalid-name
        L = shape1d(79, 3, stride=2)
        L = shape1d(L, 3, stride=2)
        L = shape1d(L, 3, stride=2)
        # pylint: enable=invalid-name
        logger.info(f'L: {L}')
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section 3 * 3* 32
        self.encoder_lin = nn.Sequential(
            nn.Linear(L * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)  # type: torch.Tensor
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, features: int):
        super().__init__()

        # pylint: disable=invalid-name
        L = shape1d(79, 3, stride=2)
        L = shape1d(L, 3, stride=2)
        L = shape1d(L, 3, stride=2)
        # pylint: enable=invalid-name
        logger.info(f'L: {L}')
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, L * 32),
            # nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, L))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 79, 3, stride=2, padding=0, output_padding=0),
        )

    def forward(self, x):
        logger.info(f'x is leaf: {x.is_leaf}')
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    """Training function"""

    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []

    for batch in dataloader:

        # Move tensor to the proper device
        batch = batch.to(device)
        # Encode data
        encoded_data = encoder(batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        logger.info(f"decoded data {decoded_data}")
        logger.info(f"batch {batch}")
        loss = loss_fn(decoded_data, batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        logger.info(f'\t partial train loss (single batch): {loss.data}')
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    """Testing function"""
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []

        for batch in dataloader:
            # Move tensor to the proper device
            batch = batch.to(device)
            # Encode data
            encoded_data = encoder(batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data
