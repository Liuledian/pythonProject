from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from nn_a4.model import LSTMClf
from nn_a4.train_lstm import SeqDataset, load_lstm_dataset, get_args


def plot_confusion(y_true, y_pred, token):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 0, 1])
    display = display.plot(cmap='Blues')
    display.figure_.suptitle(token)
    plt.savefig('{}_cm.png'.format(token))
    plt.show()


def predict(model, dataset, batch_size):
    y_pred = []
    y_true = []
    dataloader = DataLoader(dataset, batch_size)
    with torch.no_grad():
        for x, y in dataloader:
            y_true.extend(y)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            y_pred.extend(pred)

    return y_true, y_pred


def main_plot():
    args = get_args()
    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    state = ckpt['state_dict']
    model = LSTMClf(3, 310, 128, 2, 128, 0.7)
    model.load_state_dict(state)
    train_data_ls, train_label_ls, test_data_ls, test_label_ls = load_lstm_dataset(args.dir)
    # tr_dataset = SeqDataset(train_data_ls, train_label_ls)
    te_dataset = SeqDataset(test_data_ls, test_label_ls)
    y_true, y_pred = predict(model, te_dataset)
    plot_confusion(y_true, y_pred, 'lstm_cm')


if __name__ == '__main__':
    ckpt_path = './lstm_seed.ckpt'
    main_plot()
