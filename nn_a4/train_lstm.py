import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import sklearn
import logging
import argparse
from nn_a4.model import LSTMClf

LENGTHS_CLIP = [238, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
MAX_LENGTH = 238
LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LENGTHS_SUB = 3397
STEPS = 60
log_path = './log'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=100)
    return parser.parse_args()

def get_logger():
    l = logging.getLogger()
    l.setLevel(logging.NOTSET)
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    fh.setLevel(logging.NOTSET)
    sh.setLevel(logging.NOTSET)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


logger = get_logger()


class SeqDataset(Dataset):
    def __init__(self, data_ls, label_ls):
        self.data = torch.tensor(np.stack(data_ls, axis=0), dtype=torch.float)
        self.label = torch.tensor(label_ls, dtype=torch.long) + 1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


def load_lstm_dataset(dir):
    tr_data_path = os.path.join(dir, 'train_data.npy')
    te_data_path = os.path.join(dir, 'test_data.npy')
    tr_data = np.load(tr_data_path)
    te_data = np.load(te_data_path)
    scaler = sklearn.preprocessing.StandardScaler()
    tr_data = scaler.fit_transform(tr_data)
    te_data = scaler.fit_transform(te_data)

    n_tr_sub = len(tr_data) // LENGTHS_SUB
    n_te_sub = len(te_data) // LENGTHS_SUB
    logger.info('{} subjects in training dataset; {} subjects in test dataset'.format(n_tr_sub, n_te_sub))

    # Construct seq training set
    train_data_ls = []
    train_label_ls = []
    idx = 0
    for s in range(n_tr_sub):
        for c in range(15):
            length = LENGTHS_CLIP[c]
            for i in range(0, length - STEPS + 1):
                train_data_ls.append(tr_data[idx + i:idx + i + STEPS, :])
                train_label_ls.append(LABELS[c])
            idx += length

    # Construct seq test set
    test_data_ls = []
    test_label_ls = []
    idx = 0
    for s in range(n_te_sub):
        for c in range(15):
            length = LENGTHS_CLIP[c]
            for i in range(0, length - STEPS + 1):
                test_data_ls.append(te_data[idx + i:idx + i + STEPS, :])
                test_label_ls.append(LABELS[c])
            idx += length

    logger.info('train data shape {}; test data shape {};'.format(train_data_ls[0].shape, test_data_ls[0].shape))
    return train_data_ls, train_label_ls, test_data_ls, test_label_ls


def train_lstm(tr_set, te_set, epoch, batch_size, lr,  n_class, in_dim, h_dim, n_layers, cls_dim, dropout, ckpt_save_path):
    tr_loader = DataLoader(
        dataset=tr_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    te_loader = DataLoader(
        dataset=te_set,
        batch_size=batch_size,
        shuffle=True,
    )
    model = LSTMClf(n_class, in_dim, h_dim, n_layers, cls_dim, dropout)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    best_ep = 0
    for ep in range(1, epoch + 1):
        model.train()
        loss_sum = 0
        ns = 0
        for x, y in tr_loader:
            bs = len(y)
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            loss = F.cross_entropy(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * bs
            ns += bs

        te_acc, te_f1 = evaluate(model, te_loader)
        tr_acc, tr_f1 = evaluate(model, tr_loader)
        logger.info('epoch: {:>4}, loss: {:8.4f}, train acc: {:8.4f}, train f1: {:8.4f},'
                    ' eval acc: {:8.4f}, eval f1: {:8.4f}'
                    .format(ep, loss_sum / ns, tr_acc, tr_f1, te_acc, te_f1))
        if te_acc > best_acc:
            best_ep = ep
            best_acc = te_acc
            ckpt = {'state_dict': model.state_dict(), 'epoch': ep}
            torch.save(ckpt, ckpt_save_path)

    logger.info('best epoch {:>4}, best acc {:8.4f}'.format(best_ep, best_acc))


def evaluate(model, dataloader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for x, y in dataloader:
            y_true.extend(y)
            y = y.cuda()
            x = x.cuda()
            output = model(x)
            pred = torch.argmax(output, dim=1)
            y_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


def main():
    train_data_ls, train_label_ls, test_data_ls, test_label_ls = load_lstm_dataset(args.dir)
    tr_dataset = SeqDataset(train_data_ls, train_label_ls)
    te_dataset = SeqDataset(test_data_ls, test_label_ls)
    torch.manual_seed(0)
    train_lstm(tr_dataset, te_dataset, args.epoch, 16, 0.001, 3, 310, 128, 2, 128, 0.7, './lstm_seed.ckpt')


if __name__ == '__main__':
    args = get_args()
    main()
