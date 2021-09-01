
import torch
from torch import nn

from utils import init_hidden

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, seq_len: int, num_layers: int):
        """
        Initialize the model.
        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_layers=num_layers, input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.
        Args:
            input_data: (torch.Tensor): tensor of input daa
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size, self.num_layers),
                    init_hidden(input_data, self.hidden_size, self.num_layers))
        torch.nn.init.xavier_normal_(h_t)
        torch.nn.init.xavier_normal_(c_t)

        out, (h_t, c_t) = self.lstm(input_data, (h_t, c_t))
        return out, (h_t, c_t)


