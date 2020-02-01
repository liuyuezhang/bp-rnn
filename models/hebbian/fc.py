from .utils import *


class SeriesModule:
    def __init__(self, modules):
        self.modules = modules
        self.L = len(modules)

    def forward(self, x):
        for i in range(self.L):
            x = self.modules[i].forward(x)
        return x

    def backward(self, e):
        for i in reversed(range(self.L)):
            e = self.modules[i].backward(e)
        return e


class FC:
    def __init__(self, m, n, lr):
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)

        # initialization is critical
        stdv = 1. / np.sqrt(m)
        self.W = np.random.uniform(-stdv, stdv, (n, m))
        self.b = np.random.uniform(-stdv, stdv, self.out_shape)

        self.x = None
        self.y = None
        self.derive = None

        self.lr = lr
        self.alpha = 0.99
        self.r_w = np.zeros((n, m))
        self.r_b = np.zeros(self.out_shape)

    def forward(self, x):
        # input
        self.x = x.reshape(self.in_shape)
        # forward
        z = self.W @ self.x + self.b
        self.y = sigmoid(z).reshape(self.out_shape)
        # save derivative
        self.derive = self.y * (1 - self.y)
        return self.y

    def backward(self, e):
        # input
        e = e.reshape(self.out_shape)
        # calc derivative
        yx = (self.derive * self.W).T
        yw = self.derive @ self.x.T
        yb = self.derive
        # update (RMSprop)
        g_w = yw * e
        g_b = yb * e
        self.r_w = self.alpha * self.r_w + (1-self.alpha) * (g_w * g_w)
        self.r_b = self.alpha * self.r_b + (1-self.alpha) * (g_b * g_b)

        self.W -= self.lr / np.sqrt(1e-8 + self.r_w) * g_w
        self.b -= self.lr / np.sqrt(1e-8 + self.r_b) * g_b
        return yx @ e
