from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, number_of_features, number_of_classes):
        super(Encoder, self).__init__()
        self.hiddens_mu = []
        self.hiddens_logstd = []
        self.biases_mu = []
        self.biases_logstd = []
        number_of_hidden_features = []
        number_of_hidden_feature = number_of_features
        while (number_of_hidden_feature >= number_of_classes):
            number_of_hidden_features.append(number_of_hidden_feature)
            number_of_hidden_feature = number_of_hidden_feature // 2
        for number_of_hidden_feats_index in range(len(number_of_hidden_features) - 1):
            self.hiddens_mu.append(
                GCNConv(number_of_hidden_features[number_of_hidden_feats_index], number_of_hidden_features[number_of_hidden_feats_index + 1]))
            self.hiddens_logstd.append(
                GCNConv(number_of_hidden_features[number_of_hidden_feats_index], number_of_hidden_features[number_of_hidden_feats_index + 1]))
            self.biases_mu.append(nn.Parameter(torch.zeros(1)))
            self.biases_logstd.append(nn.Parameter(torch.zeros(1)))
        self.hiddens_mu = nn.ModuleList(self.hiddens_mu)
        self.hiddens_logstd = nn.ModuleList(self.hiddens_logstd)
        self.biases_mu = nn.ParameterList(self.biases_mu)
        self.biases_logstd = nn.ParameterList(self.biases_logstd)

    def forward(self, x, edge_index):
        mu = logstd = x
        for hidden_mu, hidden_logstd, bias_mu, bias_logstd in zip(self.hiddens_mu, self.hiddens_logstd, self.biases_mu, self.biases_logstd):
            mu = hidden_mu(mu, edge_index)
            logstd = hidden_logstd(logstd, edge_index)
            mu = F.relu(mu + bias_mu)
            logstd = F.relu(logstd + bias_logstd)
        mu = F.softmax(mu, dim=1)
        logstd = F.softmax(logstd, dim=1)
        return mu, logstd