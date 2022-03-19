#!/usr/bin/env python3

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, input_size: int):
        super(Encoder, self).__init__()
        self.input_size = input_size

        self.seq = nn.Sequential(
            nn.Linear(self.input_size, int(self.input_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.input_size/2), int(self.input_size/4)),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)

class Decoder(nn.Module):

    def __init__(self, decoder_input_size: int, output_size: int):
        super(Decoder, self).__init__()
        self.decoder_input_size = decoder_input_size
        self.output_size = output_size

        self.seq = nn.Sequential(
            nn.Linear(self.decoder_input_size, int(self.decoder_input_size * 2)),
            nn.ReLU(True),
            nn.Linear(int(self.decoder_input_size * 2), self.output_size),
            nn.Tanh())

    def forward(self, x: torch.Tensor):
        return self.seq(x)

class Autoencoder(nn.Module):
    
    def __init__(self, input_size: int, decoder_input_size: int, output_size: int):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size).to(device)
        self.decoder = Decoder(int(decoder_input_size), output_size).to(device)

    def forward(self, x: torch.Tensor):
        encoder_output = self.encoder(x)
        outputs = self.decoder(encoder_output)
        return outputs