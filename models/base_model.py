import torch.nn as nn
import numpy as np
import torch

class BinaryClassification(nn.Module):
    def __init__(self, n_layers):
        super(BinaryClassification, self).__init__()
        # Number of input features is 35.
        self.layer_1 = nn.Linear(35, 64)
        self.layer_2 = nn.Linear(64, 128)
        self.layer_3 = nn.Linear(128, 256)
        self.layer_4 = nn.Linear(256, 256)
        self.layer_5 = nn.Linear(256, 128)
        self.layer_6 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.batchnorm5 = nn.BatchNorm1d(128)
        self.batchnorm6 = nn.BatchNorm1d(64)

        self.double()

    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)
        #x = self.dropout_input(x)
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        #x = self.dropout_hidden(x)
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        #x = self.dropout_hidden(x)
        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        #x = self.dropout_hidden(x)
        # x = self.layer_5(x)
        # x = self.batchnorm5(x)
        # x = self.relu(x)
        #x = self.dropout_hidden(x)
        x = self.layer_6(x)
        x = self.batchnorm6(x)
        x = self.relu(x)
        #x = self.dropout_hidden(x)
        x = self.layer_7(x)
        x = self.batchnorm7(x)
        x = self.relu(x)
        #x = self.dropout_hidden(x)
        x = self.layer_out(x)

        return x


class MLP(nn.Module):
    def __init__(self, h_sizes):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            #self.hidden.append(nn.Module([nn.Linear(h_sizes[k], h_sizes[k+1]), nn.BatchNorm1d(h_sizes[k+1])]))
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.hidden.append(nn.BatchNorm1d(h_sizes[k+1]))
        self.out = nn.Linear(h_sizes[-1], 1)
        self.double()

    def forward(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        out = self.out(x)
        return out


if __name__ == '__main__':
    hiddens = 5
    start = 5
    h_sizes = [70]
    max_exp = int(hiddens/2) + start
    for i in range(hiddens):
        if i <= int(hiddens/2):
            h_sizes.append(2**(start+i))
        else:
            h_sizes.append(2**((max_exp) - (i - int(hiddens/2))))
    print(h_sizes)
    
    model = MLP(h_sizes)
    input = np.ones((16,70))
    print(model)
    print(model(torch.tensor(input)).shape)
    