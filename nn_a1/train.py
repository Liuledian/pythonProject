from mlqp import MLQP
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pickle
import math


class MinMax:

    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.model4 = None

    @staticmethod
    def split_dataset_random(X, Y):
        pos_idx = np.argwhere(Y == 1)
        n_pos = len(pos_idx)
        mid_pos = n_pos // 2

        neg_idx = np.argwhere(Y == 0)
        n_neg = len(neg_idx)
        mid_neg = n_neg // 2

        pos_perm = np.random.permutation(pos_idx)
        neg_perm = np.random.permutation(neg_idx)
        X_ls = []
        Y_ls = []
        idx_1 = np.concatenate((pos_perm[:mid_pos], neg_perm[:mid_neg]))
        X_ls.append(X[idx_1])
        Y_ls.append(Y[idx_1])
        idx_2 = np.concatenate((pos_perm[:mid_pos], neg_perm[mid_neg:]))
        X_ls.append(X[idx_2])
        Y_ls.append(Y[idx_2])
        idx_3 = np.concatenate((pos_perm[mid_pos:], neg_perm[:mid_neg]))
        X_ls.append(X[idx_3])
        Y_ls.append(Y[idx_3])
        idx_4 = np.concatenate((pos_perm[mid_pos:], neg_perm[mid_neg:]))
        X_ls.append(X[idx_4])
        Y_ls.append(Y[idx_4])

        return X_ls, Y_ls

    @staticmethod
    def split_dataset_axis(X, Y):
        pos_less_idx = np.argwhere(np.logical_and(Y == 1, X[:, 0] < 0))
        n_pos_less = len(pos_less_idx)
        pos_great_idx = np.argwhere(np.logical_and(Y == 1, X[:, 0] >= 0))
        n_pos_great = len(pos_great_idx)
        neg_less_idx = np.argwhere(np.logical_and(Y == 0, X[:, 0] < 0))
        n_neg_less = len(neg_less_idx)
        neg_great_idx = np.argwhere(np.logical_and(Y == 0, X[:, 0] >= 0))
        n_neg_great = len(neg_great_idx)
        print(n_pos_less, n_pos_great, n_neg_less, n_neg_great)

        X_ls = []
        Y_ls = []
        idx_1 = np.concatenate((pos_less_idx, neg_less_idx))
        X_ls.append(X[idx_1])
        Y_ls.append(Y[idx_1])
        idx_2 = np.concatenate((pos_less_idx, neg_great_idx))
        X_ls.append(X[idx_2])
        Y_ls.append(Y[idx_2])
        idx_3 = np.concatenate((pos_great_idx, neg_less_idx))
        X_ls.append(X[idx_3])
        Y_ls.append(Y[idx_3])
        idx_4 = np.concatenate((pos_great_idx, neg_great_idx))
        X_ls.append(X[idx_4])
        Y_ls.append(Y[idx_4])

        return X_ls, Y_ls

    def train_minmax(self, mode, in_dim, h_dims, X_tr, Y_tr, X_te, Y_te, n_epochs, lr, gamma, l2):
        if mode == 'random':
            X_ls, Y_ls = MinMax.split_dataset_random(X_tr, Y_tr)
        else:
            X_ls, Y_ls = MinMax.split_dataset_axis(X_tr, Y_tr)

        loss = [None] * 4
        acc = [None] * 4
        f1 = [None] * 4
        labels = [mode + '1', mode + '2', mode + '3', mode + '4']
        self.model1, loss[0], acc[0], f1[0] = train_mlqp(in_dim, h_dims, X_ls[0], Y_ls[0], X_te, Y_te,
                                                         n_epochs, lr, gamma, l2, labels[0])
        self.model2, loss[1], acc[1], f1[1] = train_mlqp(in_dim, h_dims, X_ls[1], Y_ls[1], X_te, Y_te,
                                                         n_epochs, lr, gamma, l2, labels[1])
        self.model3, loss[2], acc[2], f1[2] = train_mlqp(in_dim, h_dims, X_ls[2], Y_ls[2], X_te, Y_te,
                                                         n_epochs, lr, gamma, l2, labels[2])
        self.model4, loss[3], acc[3], f1[3] = train_mlqp(in_dim, h_dims, X_ls[3], Y_ls[3], X_te, Y_te,
                                                         n_epochs, lr, gamma, l2, labels[3])

        plot_metric_curve(loss, acc, f1, labels, mode)

    def forward(self, x):
        y1 = True if self.model1.forward(x) >= 0.5 else False
        y2 = True if self.model2.forward(x) >= 0.5 else False
        y3 = True if self.model3.forward(x) >= 0.5 else False
        y4 = True if self.model4.forward(x) >= 0.5 else False
        return float((y1 and y2) or (y3 and y4))


def build_dataset(path, n_te):
    data_file = open(path)
    X = []
    Y = []
    for line in data_file.readlines():
        ls = line.split()
        # Convert to float from string
        X.append([float(ls[0]), float(ls[1])])
        Y.append(float(ls[2]))

    X_tr = np.array(X)
    Y_tr = np.array(Y)

    # Generate test dataset
    X_te = []
    Y_te = []
    rs = np.linspace(0.5, 6, n_te)
    for i in range(n_te):
        r = rs[i]
        theta = r * math.pi
        X_te.append([r * math.cos(theta), r * math.sin(theta)])
        Y_te.append(1)
        X_te.append([-r * math.cos(theta), -r * math.sin(theta)])
        Y_te.append(0)
    X_te = np.array(X_te)
    Y_te = np.array(Y_te)
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


def train_mlqp(in_dim, h_dims, X_tr, Y_tr, X_te, Y_te, n_epochs, lr, gamma, l2, token):
    model = MLQP(in_dim, h_dims)
    start_lr = lr
    n_tr = len(Y_tr)
    loss_arr = []
    acc_te_arr = []
    f1_te_arr = []
    # Figure for decision boundary
    fig = plt.figure(figsize=(8, 4))
    epoch_ls = [500, 2000, n_epochs]
    for ep in range(1, n_epochs + 1):
        if ep % 500 == 0:
            lr *= gamma
        all_loss = 0
        perm = np.random.permutation(n_tr)
        for i in range(n_tr):
            p_pred = model.forward(X_tr[perm[i]])
            loss = model.compute_loss(Y_tr[perm[i]])
            all_loss += loss
            model.backward()
            model.update(lr, l2)

        # Evaluate model
        acc_te, f1_te = evaluate(model, X_te, Y_te)
        acc_tr, f1_tr = evaluate(model, X_tr, Y_tr)
        print('epoch {:>4d}; loss: {:9<.4f}; train acc: {:9<.4f};train f1 {:9<.4f}; '
              'eval acc: {:9<.4f}; eval f1: {:9<.4f}'.format(ep, all_loss/n_tr, acc_tr, f1_tr, acc_te, f1_te))
        loss_arr.append(all_loss/n_tr)
        acc_te_arr.append(acc_te)
        f1_te_arr.append(f1_te)

        # Draw boundary subplots
        if ep in epoch_ls:
            pos = epoch_ls.index(ep) + 1
            plot_subgraph(fig, [1, 3, pos], model, 'binary', 'epoch {}, lr {},{}'.format(ep, start_lr, gamma))

    plt.savefig(token + '_boundary.jpg')
    plt.show()
    return model, loss_arr, acc_te_arr, f1_te_arr


def plot_metric_curve(loss_ls, acc_ls, f1_ls, labels, token):
    fig = plt.figure(figsize=(8, 8))
    n_plot = len(loss_ls)
    ax_loss = fig.add_subplot(3, 1, 1)
    ax_loss.set_title('Loss per epoch')
    ax_acc = fig.add_subplot(3, 1, 2)
    ax_acc.set_title('Accuracy per epoch')
    ax_f1 = fig.add_subplot(3, 1, 3)
    ax_f1.set_title('F1 score per epoch')
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(n_plot):
        n_epoch = len(loss_ls[i])
        xs = np.arange(1, n_epoch + 1)
        # Plot loss
        ax_loss.plot(xs, loss_ls[i], label=labels[i], color=colors[i], linewidth=0.3)
        # Plot accuracy
        ax_acc.plot(xs, acc_ls[i], label=labels[i], color=colors[i], linewidth=0.3)
        # Plot f1
        ax_f1.plot(xs, f1_ls[i], label=labels[i], color=colors[i], linewidth=0.3)

    ax_loss.legend()
    ax_acc.legend()
    ax_f1.legend()
    plt.savefig(token + '_metric.jpg')
    plt.show()


def plot_scatter_sub(fig, position, X, Y):
    ax = fig.add_subplot(*position)
    for i in range(len(X)):
        if Y[i] == 1:
            color = 'white'
        else:
            color = 'black'
        ax.scatter(X[i, 0], X[i, 1], c=color, s=10)
    ax.set_facecolor('grey')


def plot_subgraph(fig, position, model, mode, title):
    s = 100
    ax = fig.add_subplot(*position)
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
    ax.set_title(title)


def mlqp_main():
    np.random.seed(0)
    data_path = 'two-spiral traing data(update).txt'
    X_tr, Y_tr, X_te, Y_te = build_dataset(data_path, 150)
    # Plot test dataset
    figure1 = plt.figure(figsize=(8, 8))
    plot_scatter_sub(figure1, [1, 1, 1], X_te, Y_te)
    plt.savefig('test_data.jpg')
    plt.show()
    # Train model
    lr_gamma_list = [(0.0001, 1), (0.001, 1), (0.01, 0.8)]
    labels = []
    loss_ls = []
    acc_ls = []
    f1_ls = []
    for i in range(len(lr_gamma_list)):
        lr, gamma = lr_gamma_list[i]
        token = '{}_{}'.format(lr, gamma)
        labels.append(token)
        model, loss, acc, f1 = train_mlqp(2, [60], X_tr, Y_tr, X_te, Y_te, 5000, lr, gamma, 0.001, token)
        loss_ls.append(loss)
        acc_ls.append(acc)
        f1_ls.append(f1)
        # Save model
        pickle.dump(model, open(token + '_model.pkl', 'wb'))

    plot_metric_curve(loss_ls, acc_ls, f1_ls, labels, 'mlqp')


def min_max_main():
    np.random.seed(0)
    data_path = 'two-spiral traing data(update).txt'
    X_tr, Y_tr, X_te, Y_te = build_dataset(data_path, 150)
    minmax_model = MinMax()

    # Random split and axis split
    for mode in ['random', 'axis']:
        minmax_model.train_minmax(mode, 2, [60], X_tr, Y_tr, X_te, Y_te, 3000, 0.01, 0.8, 0.001)
        figure = plt.figure(figsize=(8, 8))
        plot_subgraph(figure, [3, 2, 1], minmax_model.model1, 'binary', 'model1')
        plot_subgraph(figure, [3, 2, 2], minmax_model.model2, 'binary', 'model2')
        plot_subgraph(figure, [3, 2, 3], minmax_model.model3, 'binary', 'model3')
        plot_subgraph(figure, [3, 2, 4], minmax_model.model4, 'binary', 'model4')
        plot_subgraph(figure, [3, 2, 5], minmax_model, 'binary', 'overall')
        plt.savefig(mode + '_boundary.jpg')
        plt.show()
        pickle.dump(minmax_model, open(mode + '_model.pkl', 'wb'))


if __name__ == '__main__':
    min_max_main()
    # mlqp_main()