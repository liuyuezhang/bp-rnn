import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_first, device):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.hidden = None

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x, self.hidden = self.rnn(input, self.hidden)
        x = torch.sigmoid(self.readout(x))
        return x

    def init(self, batch_size=1):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device), \
                      torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
