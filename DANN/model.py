import torch.nn as nn
from functions import ReverseLayerF


class CNNModel(nn.Module):
    def __init__(self, in_dim, h_dims, out_dim):
        super(CNNModel, self).__init__()
        self.h_dims = h_dims
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feature = nn.Sequential()

        self.feature.add_module('f_fc0', nn.Linear(in_dim, h_dims[0]))
        self.feature.add_module('f_bn0', nn.BatchNorm1d(h_dims[0]))
        self.feature.add_module('f_relu0', nn.LeakyReLU())
        for i in range(1, len(h_dims)):
            self.feature.add_module('f_fc{}'.format(i), nn.Linear(h_dims[i-1], h_dims[i]))
            self.feature.add_module('f_bn{}'.format(i), nn.BatchNorm1d(h_dims[i]))
            self.feature.add_module('f_relu{}'.format(i), nn.LeakyReLU())

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(h_dims[-1], 64))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu1', nn.LeakyReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(64, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu2', nn.LeakyReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, out_dim))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(h_dims[-1], 64))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(64, 64))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('d_relu2', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(64, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output
