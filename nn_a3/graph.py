import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pickle


def tsne_reduce(X):
    tsne = TSNE(n_components=2)
    return tsne.fit_transform(X)


def plot_2D_scatter_domain_adapt(ax, X, labels, domains, size):
    colors = np.array(['b', 'g', 'r', 'c', 'm', 'y'])
    markers = np.array(['.', 'v', 's', '*', 'x'])
    points = tsne_reduce(X)
    print('tsne dimension reduction finished')
    labels = labels.astype(np.int32)
    domains = domains.astype(np.int32)
    for i in range(len(labels) / 100):
        ax.scatter(points[i, 0], points[i, 1], c=colors[labels[i]], marker=markers[domains[i]], s=size)


if __name__ == '__main__':
    with open('./dataset/seed/data.pkl', 'rb') as f:
        data = pickle.load(f)

    print("loading data finished")
    fig, subplots = plt.subplots(1, 2, figsize=(16, 8), squeeze=False)
    X = np.concatenate([data['sub_{}'.format(i)]['data'] for i in range(1)], axis=0)
    labels = np.concatenate([data['sub_{}'.format(i)]['label'] for i in range(1)], axis=0)
    domains = np.concatenate([np.full(len(data['sub_{}'.format(i)]['label']), i) for i in range(1)])
    plot_2D_scatter_domain_adapt(subplots[0][0], X, labels, domains, 6)
    plt.show()
