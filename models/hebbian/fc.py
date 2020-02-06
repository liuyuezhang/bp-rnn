from .utils import *


class Series:
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


class Pipeline:
    def __init__(self, modules):
        self.modules = modules
        self.L = len(modules)

        self.r = [None] * (self.L + 1)
        self.e = [None] * (self.L + 1)

    def forward(self, x, target):
        self.r[0] = x
        self.e[-1] = target
        # simultaneously forward, O(1)
        for l in range(self.L):
            if self.r[l] is not None:
                self.modules[l].forward(self.r[l])
        for l in range(self.L):
            self.r[l + 1] = self.modules[l].y
        # simultaneously backward, O(1)
        for l in range(self.L):
            if self.e[l+1] is not None:
                self.modules[l].backward(self.e[l+1])
        for l in range(self.L):
            self.e[l] = self.modules[l].d
        return self.r[-1]

    def test(self, x):
        # do not engage the loss layer
        for l in range(self.L - 1):
            x = self.modules[l].test(x)
        return x


class MSELoss:
    def __init__(self):
        # interface
        self.x = None
        self.y = None
        self.e = None
        self.d = None
        self.targets = []

    def forward(self, x):
        if x is not None:
            self.x = x
            if len(self.targets) > 0:
                self.y = np.mean(0.5 * (self.targets[0] - self.x)**2)
        return self.y

    def backward(self, target):
        self.targets.append(target)
        if self.x is not None:
            target = self.targets.pop(0)
            self.d = self.x - target
        return self.d


class FC:
    def __init__(self, m, n, lr):
        self.in_shape = (m, 1)
        self.out_shape = (n, 1)

        # initialization is critical
        stdv = 1. / np.sqrt(m)
        self.W = np.random.uniform(-stdv, stdv, (n, m))
        self.b = np.random.uniform(-stdv, stdv, self.out_shape)

        # interface
        self.x = None
        self.y = None
        self.e = None
        self.d = None

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
        self.e = e.reshape(self.out_shape)
        # calc derivative
        yx = (self.derive * self.W).T
        yw = self.derive @ self.x.T
        yb = self.derive
        # update (RMSprop)
        g_w = yw * self.e
        g_b = yb * self.e
        self.r_w = self.alpha * self.r_w + (1-self.alpha) * (g_w * g_w)
        self.r_b = self.alpha * self.r_b + (1-self.alpha) * (g_b * g_b)
        self.W -= self.lr / np.sqrt(1e-8 + self.r_w) * g_w
        self.b -= self.lr / np.sqrt(1e-8 + self.r_b) * g_b
        # output
        self.d = yx @ e
        return self.d

    def test(self, x):
        x = x.reshape(self.in_shape)
        z = self.W @ x + self.b
        y = sigmoid(z).reshape(self.out_shape)
        return y
