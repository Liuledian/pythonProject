from mlqp import MLQP
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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


def train(in_dim, h_dims, X_tr, Y_tr, X_te, Y_te, n_epochs, lr):
    model = MLQP(in_dim, h_dims)
    n_tr = len(Y_tr)
    for ep in range(1, n_epochs + 1):
        all_loss = 0
        perm = np.random.permutation(n_tr)
        for i in range(n_tr):
            p_pred = model.forward(X_tr[perm[i]])
            # print(i, p_pred, Y_tr[perm[i]])
            loss = model.compute_loss(Y_tr[perm[i]])
            all_loss += loss
            model.backward()
            model.update(lr)
        acc_te, f1_te = evaluate(model, X_te, Y_te)
        acc_tr, f1_tr = evaluate(model, X_tr, Y_tr)
        print('epoch {:>4d}; loss: {:9<.4f}; train acc: {:9<.4f};train f1 {:9<.4f}; '
              'eval acc: {:9<.4f}; eval f1: {:9<.4f}'.format(ep, all_loss/n_tr, acc_tr, f1_tr, acc_te, f1_te))


if __name__ == '__main__':
    np.random.seed(0)
    data_path = 'two-spiral traing data(update).txt'
    X_tr, Y_tr, X_te, Y_te = build_dataset(data_path, 150)
    train(2, [16, 16], X_tr, Y_tr, X_te, Y_te, 1000, 0.01)
