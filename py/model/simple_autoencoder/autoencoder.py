#!/usr/bin/env python3

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, seq_len: int, num_layers: int):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers

    def forward(self, input: torch.Tensor):
        return input

class Decoder(nn.Module):

    def __init__(self, decoder_input_size: int, seq_len: int, hidden_size: int,
                 num_layers: int, output_size: int):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder_input_size = decoder_input_size

    def forward(self, input: torch.Tensor):
        return input

class Autoencoder(nn.Module):
    
    def __init__(self, num_layers, input_size, encoder_hidden_size,
                 decoder_hidden_size, seq_len, output_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, encoder_hidden_size, seq_len,
                               num_layers).to(device)
        self.decoder = Decoder(encoder_hidden_size, seq_len, decoder_hidden_size,
                               num_layers, output_size).to(device)

    def forward(self, input: torch.Tensor):
        encoder_output, (h_t, c_t) = self.encoder(input)
        outputs = self.decoder(encoder_output)

        return outputs.unsqueeze(0)