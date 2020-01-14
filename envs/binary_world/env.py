from collections import deque
import numpy as np


def add_one(history):
    return history[-1] + 1


def fibonacci(history):
    if len(history) < 2:
        return 1
    else:
        return history[-2] + history[-1]


def xor(history):
    if len(history) < 2:
        return 1
    else:
        a = history[-2]
        b = history[-1]
        if (a > 0 and b > 0) or (a == 0 and b == 0):
            return 0
        else:
            return 1


class BinaryWorld:
    def __init__(self, d, forward, history_len=5, init_state=0):

        self.d = d
        self.history = deque([], maxlen=history_len)
        self.init_state = init_state

        self.forward = forward

    def reset(self):
        self.history.clear()
        self.history.append(self.init_state)
        return self._ob()

    def step(self):
        next_state = self._convert_state(self.forward(self.history))
        self.history.append(next_state)
        return self._ob()

    def _ob(self):
        state = self.history[-1]
        res = ('{:0'+str(self.d)+'b}').format(state)
        return np.array(list(res), dtype=np.float32).reshape((self.d, 1))

    def _convert_state(self, state):
        return int(state) % (2 ** self.d)
