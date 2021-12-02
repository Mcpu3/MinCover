from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, number_of_features, number_of_classes):
        super(GCN, self).__init__()
        self.hiddens = []
        self.biases = []
        number_of_hidden_features = []
        number_of_hidden_feature = number_of_features
        while (number_of_hidden_feature >= number_of_classes):
            number_of_hidden_features.append(number_of_hidden_feature)
            number_of_hidden_feature = number_of_hidden_feature // 2
        for number_of_hidden_feats_index in range(len(number_of_hidden_features) - 1):
            self.hiddens.append(GraphConv(number_of_hidden_features[number_of_hidden_feats_index], number_of_hidden_features[number_of_hidden_feats_index + 1]))
            self.biases.append(nn.Parameter(torch.zeros(1)))
        self.hiddens = nn.ModuleList(self.hiddens)
        self.biases = nn.ParameterList(self.biases)

    def forward(self, graph, features):
        h = features
        for hidden, bias in zip(self.hiddens, self.biases):
            h = hidden(graph, h)
            h = F.relu(h + bias)
        h = F.softmax(h, dim=1)
        return h