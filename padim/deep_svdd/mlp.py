import torch.nn as nn


class MLPNet(nn.Module):
    """A naive fully-connected backbone model for non-convolutional inputs
    """

    def __init__(self,
                 input_size: int = 100,
                 rep_dim: int = 32,
                 features_e: int = 16):
        super(MLPNet, self).__init__()

        self.rep_dim = rep_dim
        self.features_e = features_e
        self.input_size = input_size

        self.lrelu = nn.LeakyReLU()

        self.layer1 = nn.Linear(self.input_size,
                                self.features_e * 4,
                                bias=False)
        self.bn1 = nn.BatchNorm1d(self.features_e * 4, eps=1e-04, affine=False)
        self.layer2 = nn.Linear(self.features_e * 4,
                                self.features_e * 2,
                                bias=False)
        self.bn2 = nn.BatchNorm1d(self.features_e * 2, eps=1e-04, affine=False)
        self.layer3 = nn.Linear(self.features_e * 2, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.layer3(x)
        return x


class MLPNetAutoencoder(nn.Module):
    def __init__(self,
                 input_size: int = 100,
                 rep_dim: int = 32,
                 features_e: int = 16):
        super(MLPNetAutoencoder, self).__init__()

        self.rep_dim = rep_dim
        self.features_e = features_e
        self.input_size = input_size

        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # Encoder input_size -> rep_dim
        self.layer1 = nn.Linear(self.input_size,
                                self.features_e * 4,
                                bias=False)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.bn1 = nn.BatchNorm1d(self.features_e * 4, eps=1e-04, affine=False)
        self.dropout1 = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(self.features_e * 4,
                                self.features_e * 2,
                                bias=False)
        nn.init.xavier_uniform_(self.layer2.weight)
        self.bn2 = nn.BatchNorm1d(self.features_e * 2, eps=1e-04, affine=False)
        self.dropout2 = nn.Dropout(p=0.2)
        self.layer3 = nn.Linear(self.features_e * 2, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.layer3.weight)

        # Decoder rep_dim -> input_size
        self.layer4 = nn.Linear(self.rep_dim, self.features_e * 2, bias=False)
        nn.init.xavier_uniform_(self.layer4.weight)
        self.layer5 = nn.Linear(self.features_e * 2,
                                self.features_e * 4,
                                bias=False)
        nn.init.xavier_uniform_(self.layer5.weight)
        self.layer6 = nn.Linear(self.features_e * 4,
                                self.input_size,
                                bias=False)
        nn.init.xavier_uniform_(self.layer6.weight)

    def forward(self, x):
        # Encoder
        x = self.layer1(x)
        x = self.bn1(x)
        # x = self.dropout1(x)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        # x = self.dropout2(x)
        x = self.lrelu(x)
        x = self.layer3(x)

        # Decoder
        x = self.layer4(x)
        x = self.lrelu(x)
        x = self.layer5(x)
        x = self.lrelu(x)
        x = self.layer6(x)
        x = self.sigmoid(x)
        return x
