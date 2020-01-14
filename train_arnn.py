import torch
import torch.nn as nn
import numpy as np

from envs.binary_world.env import *
from models.rnn import RNN


EPOCHS = 20000
TIMESTEPS = 100
eps = 1e-4

d = 10
n = 100

device = torch.device("cuda")

env = BinaryWorld(d=d, forward=add_one)
model = RNN(input_size=d, hidden_size=n, output_size=d, device=device).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


for e in range(EPOCHS):
    total_loss = 0
    correct = 0
    state = env.reset()
    model.init()
    for t in range(TIMESTEPS):
        # forward
        state_v = torch.from_numpy(state.reshape((1, 1, -1))).float().to(device)
        pred_v = model.forward(state_v)
        next_state = env.step()

        # backward
        next_state_v = torch.from_numpy(next_state.reshape((1, 1, -1))).float().to(device)
        loss_v = criterion(pred_v, next_state_v)

        optimizer.zero_grad()
        loss_v.backward()
        optimizer.step()

        # detach
        model.detach()

        # log
        pred = pred_v.data.cpu().numpy().reshape((d, 1))
        res = np.array(pred > 0.5, dtype=np.float32)
        if np.mean((res - next_state) ** 2) < eps:
            correct += 1
        total_loss += loss_v.item()

        # update
        state = next_state

    print(e, total_loss / TIMESTEPS, correct / TIMESTEPS)
