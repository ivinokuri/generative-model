
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf

from model.autoencoder.decoder import Decoder
from model.autoencoder.encoder import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):

    def __init__(self,num_layers, input_size, encoder_hidden_size, decoder_hidden_size, seq_len, output_size):
        """
        Initialize the network.
        Args:
            config:
            input_size: (int): size of the input
        """
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, encoder_hidden_size, seq_len, num_layers).to(device)
        self.decoder = Decoder(encoder_hidden_size, seq_len, decoder_hidden_size, num_layers, output_size).to(device)

    def forward(self, encoder_input: torch.Tensor):
        """
        Forward computation. encoder_input_inputs.
        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
            return_attention: (bool): whether or not to return the attention
        """
        encoder_output, (h_t, c_t) = self.encoder(encoder_input)
        outputs = self.decoder(encoder_output)

        return outputs.unsqueeze(0)