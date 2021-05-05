import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def plot_line(xs, ys_ls, fmt_ls, label_ls, linewidth, title, save_path):
    fig = plt.figure()
    n_line = len(ys_ls)
    for i in range(n_line):
        plt.plot(xs, ys_ls[i], fmt_ls[i], label=label_ls[i], linewidth=linewidth)
    plt.legend()
    fig.suptitile(title)
    plt.savefig(save_path)
    plt.show()
    return


def tsne_reduce(X):
    tsne = TSNE(n_components=2)
    return tsne.fit_transform(X)


def plot_2D_scatter_domain_adapt(ax, points_ls, labels_ls, size):
    colors = np.array(['b', 'g', 'r', 'c', 'm', 'y'])
    markers = ['.', 'v', 's', '*', 'x']
    n_domains = len(points_ls)
    for i in range(n_domains):
        points = points_ls[i]
        labels = labels_ls[i]
        marker = markers[i]
        ax.scatter(points[:, 0], points[:, 1], c=colors[labels], s=size, marker=marker)

