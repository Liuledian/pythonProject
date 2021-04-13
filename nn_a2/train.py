from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

Y_True = None


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
        p2 = self.clf2.decision_function(X)
        # label 1
        p3 = self.clf3.decision_function(X)

        p = np.vstack([p1, p2, p3])
        pred_y = np.argmax(p, axis=0) - 1
        return pred_y

    def score(self, X, y):
        pred_y = self.predict(X)
        acc = accuracy_score(y, pred_y)
        f1 = f1_score(y, pred_y, average='macro')
        return acc, f1


class SVMMinMax:
    def __init__(self, decomp_type):
        self.svm_ls = []
        self.decomp_type = decomp_type
        for i in range(8):
            self.svm_ls.append(create_svm_grid())

    def fit(self, X, y):
        if self.decomp_type == 'random':
            self.fit_random(X, y)
        else:
            self.fit_prior(X, y)

    def fit_random(self, X, y):
        pos_idx = np.argwhere(y == 1).flatten()
        neg_idx = np.argwhere(y == 0).flatten()
        pos_idx = np.random.permutation(pos_idx)
        neg_idx = np.random.permutation(neg_idx)
        n_pos = len(pos_idx)
        mid_pos = n_pos // 2
        n_neg = len(neg_idx)
        d = n_neg // 4

        self.fit_combine(X, y, pos_idx[:mid_pos], pos_idx[mid_pos:],
                         neg_idx[:d], neg_idx[d:2*d], neg_idx[2*d:3*d], neg_idx[3*d:])

    def fit_prior(self, X, y):
        pass

    def fit_combine(self, X, y, p1, p2, n1, n2, n3, n4):
        idx = [None] * 8
        idx[0] = np.concatenate([p1, n1])
        idx[1] = np.concatenate([p1, n2])
        idx[2] = np.concatenate([p1, n3])
        idx[3] = np.concatenate([p1, n4])
        idx[4] = np.concatenate([p2, n1])
        idx[5] = np.concatenate([p2, n2])
        idx[6] = np.concatenate([p2, n3])
        idx[7] = np.concatenate([p2, n4])
        for i in range(8):
            print("Training {}/8 sub svm".format(i))
            self.svm_ls[i].fit(X[idx[i], :], y[idx[i]])

    def decision_function(self, X):
        up_half = []
        for i in range(0, 4):
            up_half.append(self.svm_ls[i].decision_function(X))
        up_res = np.min(np.vstack(up_half), axis=0)

        down_half = []
        for i in range(4, 8):
            down_half.append(self.svm_ls[i].decision_function(X))
        down_res = np.min(np.vstack(down_half), axis=0)

        res = np.maximum(up_res, down_res)
        return res


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
    svm_grid = GridSearchCV(svm, param_grid, n_jobs=-1, cv=4, return_train_score=True)
    return svm_grid


def main_svm_orv():
    tr_data, tr_label, te_data, te_label = load_dataset('/Users/unity/Downloads/data_hw2', 'standard')
    svm_orv = SvmOvr(create_svm_grid(), create_svm_grid(), create_svm_grid())
    svm_orv.fit(tr_data, tr_label)
    acc, f1 = svm_orv.score(te_data, te_label)
    print('Test  Accuracy: {}, F1: {}'.format(acc, f1))
    acc, f1 = svm_orv.score(tr_data, tr_label)
    print('Train Accuracy: {}, F1: {}'.format(acc, f1))

    print(svm_orv.clf1.cv_results_)
    print(svm_orv.clf2.cv_results_)
    print(svm_orv.clf3.cv_results_)


def main_svm_minmax_random():
    tr_data, tr_label, te_data, te_label = load_dataset('/Users/unity/Downloads/data_hw2', 'standard')
    svm_orv = SvmOvr(SVMMinMax('random'), SVMMinMax('random'), SVMMinMax('random'))
    svm_orv.fit(tr_data, tr_label)
    acc, f1 = svm_orv.score(te_data, te_label)
    print('Test  Accuracy: {}, F1: {}'.format(acc, f1))
    acc, f1 = svm_orv.score(tr_data, tr_label)
    print('Train Accuracy: {}, F1: {}'.format(acc, f1))


def main_svm_minmax_prior():
    tr_data, tr_label, te_data, te_label = load_dataset('/Users/unity/Downloads/data_hw2', 'standard')
    global Y_True
    Y_True = tr_label
    svm_orv = SvmOvr(SVMMinMax('prior'), SVMMinMax('prior'), SVMMinMax('prior'))
    svm_orv.fit(tr_data, tr_label)
    acc, f1 = svm_orv.score(te_data, te_label)
    print('Test  Accuracy: {}, F1: {}'.format(acc, f1))
    acc, f1 = svm_orv.score(tr_data, tr_label)
    print('Train Accuracy: {}, F1: {}'.format(acc, f1))

if __name__ == '__main__':
    '''
       -1 12320
        0 12144
        1 12903
          37367
    '''
    main_svm_minmax()

