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


