import torch
import torch.nn as nn


class LSTMClf(nn.Module):
    def __init__(self, n_class=3, in_dim=310, h_dim=128, n_layers=2, cls_dim=128, dropout=0.7):
        super(LSTMClf, self).__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=h_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.cls = nn.Sequential(
            nn.Linear(h_dim, cls_dim),
            nn.BatchNorm1d(cls_dim),
            nn.LeakyReLU(),
            nn.Linear(cls_dim, n_class),
        )
        self.h0 = nn.Parameter(torch.zeros(n_layers, h_dim))
        self.c0 = nn.Parameter(torch.zeros(n_layers, h_dim))
        nn.init.kaiming_uniform_(self.h0)
        nn.init.kaiming_uniform_(self.c0)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = self.h0.expand(batch_size, self.n_layers, self.h_dim)
        h0 = h0.transpose(0, 1)
        c0 = self.c0.expand(batch_size, self.n_layers, self.h_dim)
        c0 = c0.transpose(0, 1)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.cls(hn[-1])
        return output