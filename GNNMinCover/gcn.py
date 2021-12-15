from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, number_of_x, number_of_classes):
        super(GCN, self).__init__()
        self.layers, self.biases = [], []
        number_of_layers = []
        number_of_layer = number_of_x
        while (number_of_layer >= number_of_classes):
            number_of_layers.append(number_of_layer)
            number_of_layer //= 2
        for index in range(len(number_of_layers) - 1):
            self.layers.append(GraphConv(number_of_layers[index], number_of_layers[index + 1]))
            self.biases.append(nn.Parameter(torch.zeros(1)))
        self.layers = nn.ModuleList(self.layers)
        self.biases = nn.ParameterList(self.biases)

    def forward(self, data, x):
        for hidden, bias in zip(self.layers, self.biases):
            x = hidden(data, x)
            x = F.relu(x + bias)
        x = F.softmax(x, dim=1)
        return x
