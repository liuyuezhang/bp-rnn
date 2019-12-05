import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, device):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.hidden = torch.zeros(num_layers, 1, hidden_size).to(device)

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first)

    def forward(self, input):
        x, self.hidden = self.rnn(input, self.hidden)
        return x

    def init(self):
        self.hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device), \
                      torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
