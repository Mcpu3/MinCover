import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Encoder(nn.Module):
    def __init__(self, number_of_x, number_of_classes):
        super(Encoder, self).__init__()
        self.layers_mu, self.layers_logstd, self.biases_mu, self.biases_logstd = [], [], [], []
        number_of_layers_x = []
        number_of_layer_x = number_of_x
        while (number_of_layer_x >= number_of_classes):
            number_of_layers_x.append(number_of_layer_x)
            number_of_layer_x = number_of_layer_x // 2
        for index in range(len(number_of_layers_x) - 1):
            self.layers_mu.append(GCNConv(number_of_layers_x[index], number_of_layers_x[index + 1]))
            self.layers_logstd.append(GCNConv(number_of_layers_x[index], number_of_layers_x[index + 1]))
            self.biases_mu.append(nn.Parameter(torch.zeros(1)))
            self.biases_logstd.append(nn.Parameter(torch.zeros(1)))
        self.layers_mu = nn.ModuleList(self.layers_mu)
        self.layers_logstd = nn.ModuleList(self.layers_logstd)
        self.biases_mu = nn.ParameterList(self.biases_mu)
        self.biases_logstd = nn.ParameterList(self.biases_logstd)

    def forward(self, x, edge_index):
        mu, logstd = x, x
        for layer_mu, layer_logstd, bias_mu, bias_logstd in zip(self.layers_mu, self.layers_logstd, self.biases_mu, self.biases_logstd):
            mu = layer_mu(mu, edge_index)
            logstd = layer_logstd(logstd, edge_index)
            mu = F.relu(mu + bias_mu)
            logstd = F.relu(logstd + bias_logstd)
        mu = F.softmax(mu, dim=1)
        logstd = F.softmax(logstd, dim=1)
        return mu, logstd
