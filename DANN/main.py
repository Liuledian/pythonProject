import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sklearn.metrics
from model import CNNModel
import pickle
import logging


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


log_path = './log'
logger = get_logger()


class SeedDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32)
        # convert label to start from 0
        self.label = torch.tensor(label + 1, dtype=torch.long)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


def evaluate_dann(model, dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, label in dataloader:
            y_true.extend(label.detach().cpu().numpy())
            class_output, _ = model(data, 0)
            y_pred_torch = torch.argmax(class_output, dim=1)
            y_pred.extend(y_pred_torch.detach().cpu().numpy())

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    return accuracy, f1_score


def train_dann(dataset_source, dataset_target, n_epoch, batch_size, in_dim, h_dims, out_dim, ckpt_save_path):
    cuda = False
    lr = 1e-3
    l_d = 0.1

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
   )

    model = CNNModel(in_dim, h_dims, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    if cuda:
        model = model.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in model.parameters():
        p.requires_grad = True

    # training
    best_acc = 0.0
    best_ep = 0
    tr_acc_ls = []
    te_acc_ls = []
    loss_ls = []
    for epoch in range(n_epoch):
        model.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
        loss_sum = 0.0
        n_s = 0
        for i in range(len_dataloader):
            # Compute reverse layer parameter alpha
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_s, label_s = data_source_iter.next()
            batch_size_s = len(label_s)
            n_s += batch_size_s
            domain_label = torch.zeros(batch_size_s).long()

            if cuda:
                data_s = data_s.cuda()
                label_s = label_s.cuda()
                domain_label = domain_label.cuda()

            class_output, domain_output = model(input_data=data_s, alpha=alpha)
            loss_c = loss_class(class_output, label_s)
            loss_ds = loss_domain(domain_output, domain_label)

            # training model using target data
            data_t, _ = data_target_iter.next()
            batch_size_t = len(data_t)
            domain_label = torch.ones(batch_size_t).long()

            if cuda:
                data_t = data_t.cuda()
                domain_label = domain_label.cuda()
            _, domain_output = model(input_data=data_t, alpha=alpha)
            loss_dt = loss_domain(domain_output, domain_label)

            # Compute overall loss and backprop
            loss = loss_c + l_d * (loss_dt + loss_ds)
            loss_sum += loss.item() * batch_size_s

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # logger.info('epoch: {:>4}, [iter: {:>4} / all {:>4}], loss {:8.4f}, '
            #             'loss_c: {:8.4f}, loss_ds: {:8.4f}, loss_dt: {:8.4f}\n'
            #             .format(epoch, i+1, len_dataloader, loss.item(), loss_c.item(), loss_ds.item(), loss_dt.item()))

        tr_acc, tr_f1 = evaluate_dann(model, dataset_source, batch_size)
        te_acc, te_f1 = evaluate_dann(model, dataset_target, batch_size)
        tr_acc_ls.append(tr_acc)
        te_acc_ls.append(te_acc)
        loss_ls.append(loss_sum)
        # If find a better result, save the model
        if te_acc > best_acc:
            best_acc = te_acc
            best_ep = epoch
            checkpoint = {"epoch": epoch, "state_dict": model.state_dict()}
            torch.save(checkpoint, ckpt_save_path + '.ckpt')

        logger.info('epoch: {:>4}, loss: {:8.4f}, train acc: {:8.4f}, train f1: {:8.4f},'
                    ' eval acc: {:8.4f}, eval f1: {:8.4f}'
                    .format(epoch, loss_sum, tr_acc, tr_f1, te_acc, te_f1))

    logger.info('='*10)
    logger.info('best epoch: {:>4}, best acc: {:8.4f}'.format(best_ep, best_acc))
    pickle.dump(tr_acc_ls, open(ckpt_save_path + '.tracc', 'wb'))
    pickle.dump(te_acc_ls, open(ckpt_save_path + '.teacc', 'wb'))
    pickle.dump(loss_ls, open(ckpt_save_path + '.loss', 'wb'))


def main():
    with open('./dataset/seed/data.pkl', 'rb') as f:
        data = pickle.load(f)
        subject_ls = list(data.keys())
        # Iterate over all folds
        for sub in subject_ls:
            # Construct dataset
            target_data = data[sub]
            dataset_target = SeedDataset(target_data['data'], target_data['label'])
            dataset_source = SeedDataset(
                data=np.concatenate([data[sub]['data'] for s in subject_ls if s != sub], axis=0),
                label=np.concatenate([data[sub]['label'] for s in subject_ls if s != sub], axis=0)
            )
            logger.info(sub + ' dataset_source: {}, dataset_target: {} built!'
                        .format(len(dataset_source), len(dataset_target)))
            # Train this fold
            train_dann(dataset_source, dataset_target, 1000, 32, 310, [128, 128], 3, './models/{}'.format(sub))


if __name__ == '__main__':
    manual_seed = 0
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    main()

