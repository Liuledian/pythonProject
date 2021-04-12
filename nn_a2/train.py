from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os


class SvmOvr:
    def __init__(self, clf1, clf2, clf3):
        self.clf1 = clf1
        self.clf2 = clf2
        self.clf3 = clf3
        self.n_class = 3

    @staticmethod
    def build_ovr_label(label, one):
        print("Build OVR dataset {}".format(one))
        bool_idx = label == one
        y = np.zeros_like(label)
        y[bool_idx] = 1
        return y

    def fit(self, tr_data, tr_label):
        y1 = SvmOvr.build_ovr_label(tr_label, -1)
        self.clf1.fit(tr_data, y1)
        y2 = SvmOvr.build_ovr_label(tr_label, 0)
        self.clf2.fit(tr_data, y2)
        y3 = SvmOvr.build_ovr_label(tr_label, 1)
        self.clf3.fit(tr_data, y3)

    def predict(self, X):
        # label -1
        p1 = self.clf1.decision_function(X)
        # label 0
        p2 = self.clf2.decision_funtion(X)
        # label 1
        p3 = self.clf3.decision_function(X)

        p = np.vstack([p1, p2, p3])
        pred_y = np.argmax(p, axis=0) - 1
        return pred_y

    def score(self, X, y):
        pred_y = self.predict(X)
        acc = accuracy_score(y, pred_y)
        f1 = f1_score(y, pred_y, average='macro')
        print('Accuracy: {}, F1: {}'.format(acc, f1))
        return acc, f1


def load_dataset(dir, scale):
    tr_data_path = os.path.join(dir, 'train_data.npy')
    tr_label_path = os.path.join(dir, 'train_label.npy')
    te_data_path = os.path.join(dir, 'test_data.npy')
    te_label_path = os.path.join(dir, 'test_label.npy')
    tr_data = np.load(tr_data_path)
    tr_label = np.load(tr_label_path)
    te_data = np.load(te_data_path)
    te_label = np.load(te_label_path)
    if scale == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    tr_data = scaler.fit_transform(tr_data)
    te_data = scaler.fit_transform(te_data)
    return tr_data, tr_label, te_data, te_label


def create_svm_grid():
    c_range = np.logspace(-5, 15, 11, base=2)
    param_grid = {'C': c_range}
    # According to sklearn docs dual is better False
    svm = LinearSVC(dual=False)
    svm_grid = GridSearchCV(svm, param_grid, n_jobs=-1, verbose=4, cv=4, return_train_score=True)
    return svm_grid


if __name__ == '__main__':
    tr_data, tr_label, te_data, te_label = load_dataset('/Users/unity/Downloads/data_hw2', 'standard')
    svm_orv = SvmOvr(create_svm_grid(), create_svm_grid(), create_svm_grid())
    svm_orv.fit(tr_data, tr_label)
    svm_orv.score(te_data, te_label)
