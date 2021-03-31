import numpy as np


class MLQP():
    def __init__(self, in_dim, hidden_dims):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.n_hidden = len(self.hidden_dims)
        self.out_dim = 1

        # Create and initialize model parameters
        self.Wl_ls = [None] * (self.n_hidden + 2)
        self.Wq_ls = [None] * (self.n_hidden + 2)
        self.b_ls = [None] * (self.n_hidden + 2)
        # Kaiming initialization
        self.Wl_ls[1] = np.random.randn(in_dim, hidden_dims[0]) / np.sqrt(2 / in_dim)
        self.Wq_ls[1] = np.random.randn(in_dim, hidden_dims[0]) / np.sqrt(2 / in_dim)
        self.b_ls[1] = np.zeros([1, hidden_dims[0]])
        for i in range(2, self.n_hidden + 1):
            self.Wl_ls[i] = np.random.randn(hidden_dims[i - 2], hidden_dims[i - 1]) / np.sqrt(2/hidden_dims[i - 2])
            self.Wq_ls[i] = np.random.randn(hidden_dims[i - 2], hidden_dims[i - 1]) / np.sqrt(2 / hidden_dims[i - 2])
            self.b_ls[i] = np.zeros([1, hidden_dims[i - 1]])

        self.Wl_ls[-1] = np.random.randn(hidden_dims[-1], self.out_dim) / np.sqrt(2 / hidden_dims[-1])
        self.Wq_ls[-1] = np.random.randn(hidden_dims[-1], self.out_dim) / np.sqrt(2 / hidden_dims[-1])
        self.b_ls[-1] = np.zeros([1, self.out_dim])

        # Present network dims
        for i in range(1, self.n_hidden + 2):
            print('Wl_{0}: {1}; Wq_{0}: {2}, b_{0}: {3}'.format(
                i + 1, self.Wl_ls[i].shape, self.Wq_ls[i].shape, self.b_ls[i].shape
            ))

        # Create placeholder for intermediate output
        self.zs = [None] * (self.n_hidden + 2)
        self.xs = [None] * (self.n_hidden + 2)
        # Create placeholder for local gradient delta_k
        self.deltas = [None] * (self.n_hidden + 2)
        self.loss = None
        self.one_hot_y = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_dot(z):
        return MLQP.sigmoid(z) * (1 - MLQP.sigmoid(z))

    def forward(self, x):
        # x shape (1, in_dim)
        self.xs[0] = x.reshape(1, self.in_dim)
        for i in range(1, self.n_hidden + 2):
            self.zs[i] = np.square(self.xs[i - 1]) @ self.Wq_ls[i] \
                + self.xs[i - 1] @ self.Wl_ls[i] \
                + self.b_ls[i]
            self.xs[i] = MLQP.sigmoid(self.zs[i])
        assert self.xs[-1].shape[1] == 1
        return self.xs[-1].item()

    def compute_loss(self, y_true):
        p_pred = self.xs[-1]
        p_pred = np.hstack([p_pred, 1 - p_pred])
        self.one_hot_y = np.zeros([1, 2])
        if y_true == 1:
            self.one_hot_y[0, 0] = 1
        else:
            self.one_hot_y[0, 1] = 1
        self.loss = np.sum(-self.one_hot_y * np.log2(p_pred))
        return self.loss

    def backward(self):
        p_pred = self.xs[-1]
        t = np.hstack([-1 / p_pred, 1 / (1 - p_pred)])
        self.deltas[-1] = np.sum(t * self.one_hot_y) * MLQP.sigmoid_dot(self.zs[-1])
        for i in range(self.n_hidden, 0, -1):
            self.deltas[i] = (self.Wq_ls[i + 1] * 2 * self.xs[i].T + self.Wl_ls[i + 1]) \
                             @ self.deltas[i + 1] \
                             * MLQP.sigmoid_dot(self.zs[i]).T

    def update(self, lr, l2):
        for i in range(1, self.n_hidden + 2):
            dWl = np.tile(self.xs[i - 1].T, (1, self.Wl_ls[i].shape[1])) * self.deltas[i].T
            dWq = np.tile(np.square(self.xs[i - 1].T), (1, self.Wq_ls[i].shape[1])) * self.deltas[i].T
            db = self.deltas[i].T
            self.Wl_ls[i] -= lr * (dWl + l2 * self.Wl_ls[i])
            self.Wq_ls[i] -= lr * (dWq + l2 * self.Wq_ls[i])
            self.b_ls[i] -= lr * db

