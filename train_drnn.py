from envs.binary_world.env import *
from models.hebbian.rnn import DynamicRNN


EPOCHS = 20000
TIMESTEPS = 100
eps = 1e-4

d = 10

env = BinaryWorld(d=d, forward=add_one)
model = DynamicRNN(d=d, n=100, alpha=0.5, lr=1e-2)

for e in range(EPOCHS):
    total_loss = 0
    correct = 0
    state = env.reset()
    model.reset()
    for t in range(TIMESTEPS):
        # forward
        pred = model.forward(state)
        next_state = env.step()

        # backward
        error = pred - next_state
        model.backward(error)

        # log
        res = np.array(pred > 0.5, dtype=np.float32)
        if np.mean((res - next_state) ** 2) < eps:
            correct += 1
        total_loss += np.mean(error ** 2)

        # update
        state = next_state

    print(e, total_loss / TIMESTEPS, correct / TIMESTEPS)
