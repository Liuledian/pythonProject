from mlqp import MLQP
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pickle


def build_dataset(path, n_train_samples):
    data_file = open(path)
    X = []
    Y = []
    for line in data_file.readlines():
        ls = line.split()
        # Convert to float from string
        X.append([float(ls[0]), float(ls[1])])
        Y.append(float(ls[2]))
    n = len(X)
    perm = np.random.permutation(n)
    X = np.array(X)
    x1_max = np.max(X[:, 0])
    x1_min = np.min(X[:, 0])
    x2_max = np.max(X[:, 1])
    x2_min = np.min(X[:, 1])
    print('x1 [{}, {}]; x2 [{}, {}]'.format(x1_min, x1_max, x2_min, x2_max))
    Y = np.array(Y)
    X_tr = X[perm[:n_train_samples]]
    Y_tr = Y[perm[:n_train_samples]]
    X_te = X[perm[n_train_samples:]]
    Y_te = Y[perm[n_train_samples:]]
    return X_tr, Y_tr, X_te, Y_te


def evaluate(model, X, Y):
    n_sample = len(X)
    y_pred = []
    for i in range(n_sample):
        p_pred = model.forward(X[i])
        y_pred.append(1 if p_pred >= 0.5 else 0)
    acc = accuracy_score(Y, y_pred)
    f1 = f1_score(Y, y_pred)
    return acc, f1


def train(in_dim, h_dims, X_tr, Y_tr, X_te, Y_te, n_epochs, lr, gamma, l2):
    model = MLQP(in_dim, h_dims)
    n_tr = len(Y_tr)
    for ep in range(1, n_epochs + 1):
        if ep % 500 == 0:
            lr *= gamma
        all_loss = 0
        perm = np.random.permutation(n_tr)
        for i in range(n_tr):
            p_pred = model.forward(X_tr[perm[i]])
            # print(i, p_pred, Y_tr[perm[i]])
            loss = model.compute_loss(Y_tr[perm[i]])
            all_loss += loss
            model.backward()
            model.update(lr, l2)
        acc_te, f1_te = evaluate(model, X_te, Y_te)
        acc_tr, f1_tr = evaluate(model, X_tr, Y_tr)
        print('epoch {:>4d}; loss: {:9<.4f}; train acc: {:9<.4f};train f1 {:9<.4f}; '
              'eval acc: {:9<.4f}; eval f1: {:9<.4f}'.format(ep, all_loss/n_tr, acc_tr, f1_tr, acc_te, f1_te))
    return model


def plot_graph(model, mode):
    s = 100
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lin = np.linspace(-6, 6, s)
    hm = np.zeros((s, s))
    for i in range(s):
        for j in range(s):
            x = np.array([lin[i], lin[j]])
            p_pred = model.forward(x)
            if mode == 'grey':
                hm[s - j - 1, i] = p_pred
            else:
                hm[s - j - 1, i] = 1 if p_pred >= 0.5 else 0

    ax.imshow(hm, cmap=plt.cm.gray)
    ax.set_xticks(np.linspace(0, s - 1, 13))
    ax.set_yticks(np.linspace(0, s - 1, 13))
    ax.set_xticklabels(np.arange(-6, 7, 1))
    ax.set_yticklabels(np.arange(6, -7, -1))
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    data_path = 'two-spiral traing data(update).txt'
    X_tr, Y_tr, X_te, Y_te = build_dataset(data_path, 150)
    model = train(2, [16, 16], X_tr, Y_tr, X_te, Y_te, 10, 0.01, 0.8, 0.001)
    plot_graph(model, 'binary')
    pickle.dump(model, open('model.pkl', 'wb'))
