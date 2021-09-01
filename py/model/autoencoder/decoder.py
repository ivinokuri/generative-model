
import torch
from torch import nn

from utils import init_hidden

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Decoder(nn.Module):
    def __init__(self, decoder_input_size, seq_len, hidden_size, num_layers, output_size):
        """
        Initialize the network.
        Args:
            config:
        """
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder_input_size = decoder_input_size # number of features of lstm (it wiil be an output size of encoded features)
        self.lstm = nn.LSTM(decoder_input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input: torch.Tensor):
        """
        Forward pass
        Args:
            _:
            :param decoder_input:
        """
        h_t, c_t = (init_hidden(decoder_input, self.hidden_size, self.num_layers),
                    init_hidden(decoder_input, self.hidden_size, self.num_layers))
        # decoder_input = decoder_input.repeat(1, self.seq_len, 1)
        # for t in range(self.seq_len):
        #     inp = decoder_input[:, t].unsqueeze(0).unsqueeze(2)
        #     lstm_out, (h_t, c_t) = self.lstm(inp, (h_t, c_t))
        decoder_output, _ = self.lstm(decoder_input, (h_t, c_t))
        return self.fc(decoder_output.squeeze(0))
