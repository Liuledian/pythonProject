import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
import sklearn

LENGTHS_CLIP = [238, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
MAX_LENGTH = 238
LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LENGTHS_SUB = 3397


class SeqDataset(Dataset):
    def __init__(self, data, label, lengths):
        self.data = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label + 1, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.lengths[item]


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
    print('{} subjects in training dataset; {} subjects in test dataset'.format(n_tr_sub, n_te_sub))

    # Construct sequence training set
    n_fea = len(tr_data[0])
    max_time = MAX_LENGTH
    n_tr_seq = n_tr_sub * 15
    tr_time_data = np.zeros((n_tr_seq, max_time, n_fea))
    tr_time_label = np.tile(np.array(LABELS), (n_tr_sub,))
    tr_lengths = np.zeros(n_tr_seq)
    idx = 0
    for i in range(n_tr_seq):
        length = LENGTHS_CLIP[i % 15]
        tr_lengths[i] = length
        tr_time_data[i, :length, :] = tr_data[idx:idx + length, :]
        idx += length
    print(tr_time_data.shape)

    # Construct sequence test set
    n_te_seq = n_te_sub * 15
    te_time_data = np.zeros((n_te_seq, max_time, n_fea))
    te_time_label = np.tile(np.array(LABELS), (n_te_sub,))
    te_lengths = np.zeros(n_te_seq)
    idx = 0
    for i in range(n_te_seq):
        length = LENGTHS_CLIP[i % 15]
        te_lengths[i] = length
        te_time_data[i, :length, :] = tr_data[idx:idx + length, :]
        idx += length
    print(te_time_data.shape)

    return tr_time_data, tr_time_label, tr_lengths, te_time_data, te_time_label, te_lengths


def train_lstm(tr_set, te_set, epoch, batch_size, lr, in_dim, h_dim, n_layers, dropout, ckpt_save_path):
    dataloader = DataLoader(
        dataset=tr_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    lstm = torch.nn.LSTM(input_size=in_dim, hidden_size=h_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
    optimizer = torch.optim.Adam()
    best_acc = 0
    best_ep = 0



def main():
    tr_data, tr_label, tr_lengths, te_data, te_label, te_lengths = load_lstm_dataset('/Users/unity/Downloads/data_hw2')
    tr_set = SeqDataset(tr_data, tr_label, tr_lengths)
    te_set = SeqDataset(te_data, te_label, te_lengths)

