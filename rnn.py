import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity, batch_first, device):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.hidden = torch.zeros(num_layers, 1, hidden_size).to(device)

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first,
                          nonlinearity=nonlinearity)

    def forward(self, input):
        x, self.hidden = self.rnn(input, self.hidden)
        return x

    def init(self):
        self.hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
