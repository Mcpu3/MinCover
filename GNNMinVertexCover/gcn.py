from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, number_of_x, number_of_classes):
        super(GCN, self).__init__()
        self.layers = self.biases = []
        number_of_layers_x = []
        number_of_layer_x = number_of_x
        while (number_of_layer_x >= number_of_classes):
            number_of_layers_x.append(number_of_layer_x)
            number_of_layer_x = number_of_layer_x // 2
        for index in range(len(number_of_layers_x) - 1):
            self.layers.append(GraphConv(number_of_layers_x[index], number_of_layers_x[index + 1]))
            self.biases.append(nn.Parameter(torch.zeros(1)))
        self.layers = nn.ModuleList(self.layers)
        self.biases = nn.ParameterList(self.biases)

    def forward(self, graph, x):
        h = x
        for hidden, bias in zip(self.layers, self.biases):
            h = hidden(graph, h)
            h = F.relu(h + bias)
        h = F.softmax(h, dim=1)
        return h
