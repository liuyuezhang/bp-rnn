import torch
import torch.nn as nn
import numpy as np

from rnn import RNN
from lstm import LSTM


def sample_data(n):
    res = np.random.rand(n)
    res = np.array(res >= 0.5, dtype=np.int)
    return res.copy(), res[-1] * 2 - 1


TRAIN_SAMPLES = 100000
TRAIN_LENGTH = 2

device = torch.device('cuda')

# rnn = RNN(1, 100, 1, 'tanh', True, device).to(device)
rnn = LSTM(1, 50, 1, True, device).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=1e-3)

# train
for i in range(TRAIN_SAMPLES):
    # data
    x, y = sample_data(np.random.randint(TRAIN_LENGTH, TRAIN_LENGTH+1))
    x = x.reshape([1, -1, 1])

    # forward
    rnn.init()
    x_v = torch.from_numpy(x).float().to(device)
    output_v = rnn(x_v)
    pred_v = output_v[0, 0, -1]
    y_v = torch.from_numpy(np.array([y])).float().to(device)
    loss_v = criterion(pred_v, y_v)

    # backward
    optimizer.zero_grad()
    loss_v.backward()
    optimizer.step()
    print(loss_v.item())
