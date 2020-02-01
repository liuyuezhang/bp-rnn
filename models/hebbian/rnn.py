from .utils import *


class RNN:
    def __init__(self, d, n, alpha=0.5, lr=0.1):
        self.U = np.random.rand(n, d)
        self.V = np.random.rand(d, n)
        self.W = np.random.rand(n, n)

        self.alpha = alpha
        self.f = sigmoid

        self.d = d
        self.n = n
        self.lr = lr

        self.u = np.zeros((n, 1))
        self.s = None
        self.y = None

    def forward(self, x):
        h = self.U @ x
        self.s = self.f(self.u)  # save the current activation for backward computation
        self.u = (1 - self.alpha) * self.u + self.alpha * self.W @ self.s + self.alpha * h
        self.y = sigmoid(self.V @ self.u)
        return self.y

    def backward(self, error):
        # delta = self.alpha * self.V.T @ (error * self.y * (1 - self.y)) @ self.s.T
        delta = self.alpha * self.V.T @ error @ self.s.T
        self.W -= self.lr * delta

    def reset(self):
        self.u = np.zeros((self.n, 1))
        self.s = None
