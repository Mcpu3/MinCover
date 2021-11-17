from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
    dataset = Dataset()
    model = VGAE(Encoder(dataset.n_feats, 32, 16))
    model.train()
    model = train(dataset, model)
    torch.save(model.state_dict(), './models/model.pth')
    model.eval()
    test(dataset, model)


class Dataset():
    def __init__(self):
        self.n_feats = 64
        self.dataset = []
        nodes = pd.read_csv('./dataset/nodes.csv')
        edges = pd.read_csv('./dataset/edges.csv')
        nodes_group = nodes.groupby('g_id')
        edges_group = edges.groupby('g_id')
        n_nodes_dict = {}
        for g_id in nodes_group.groups:
            nodes_of_id = nodes_group.get_group(g_id)
            n_nodes = nodes_of_id.to_numpy().shape[0]
            n_nodes_dict[g_id] = n_nodes
        for g_id in nodes_group.groups:
            src = torch.empty(0, dtype=torch.int64)
            dst = torch.empty(0, dtype=torch.int64)
            if g_id in edges_group.groups.keys():
                edges_of_id = edges_group.get_group(g_id)
                src = torch.from_numpy(edges_of_id['src'].to_numpy())
                dst = torch.from_numpy(edges_of_id['dst'].to_numpy())
            x = torch.rand([n_nodes, self.n_feats], dtype=torch.float32)
            edge_index = torch.stack((src, dst))
            data = Data(x, edge_index)
            self.dataset.append(data)
        n_dataset = len(self.dataset)
        n_train = int(n_dataset * 0.8)
        self.n_train = n_train

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class Encoder(nn.Module):
    def __init__(self, n_feats, n_hidden_feats, n_classes):
        super(Encoder, self).__init__()
        self.hidden = GCNConv(n_feats, n_hidden_feats)
        self.hidden_mu = GCNConv(n_hidden_feats, n_classes)
        self.hidden_logstd = GCNConv(n_hidden_feats, n_classes)

    def forward(self, x, edge_index):
        x = self.hidden(x, edge_index)
        x = F.relu(x)
        mu = self.hidden_mu(x, edge_index)
        logstd = self.hidden_logstd(x, edge_index)
        return mu, logstd


def train(dataset, model):
    n_train = dataset.n_train
    train_dataset = dataset[:n_train]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with SummaryWriter() as writer:
        for e in tqdm(range(96)):
            aucs = np.array([])
            aps = np.array([])
            with tqdm(train_dataset) as pbar:
                for data in pbar:
                    x, edge_index = data['x'], data['edge_index']
                    mu, logstd = model.encoder(x, edge_index)
                    z = model.encode(x, edge_index)
                    neg_edge_index = negative_sampling(edge_index, z.size(0))
                    loss = model.recon_loss(z, edge_index, neg_edge_index) + \
                        model.kl_loss(mu, logstd)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    model.eval()
                    auc, ap = model.test(z, edge_index, neg_edge_index)
                    model.train()
                    aucs = np.append(aucs, auc)
                    aps = np.append(aps, ap)
                    pbar.set_postfix_str(
                        'epoch: {}, loss: {:.3f}, auc: {:.3f}, ap: {:.3f}'.format(e, loss, np.average(aucs), np.average(aps)))
            writer.add_scalar('Loss/train', loss, e)
            writer.add_scalar('AUC/train', np.average(aucs), e)
            writer.add_scalar('AP/train', np.average(aps), e)
    return model


def test(dataset, model):
    n_train = dataset.n_train
    test_dataset = dataset[n_train:]
    aucs = np.array([])
    aps = np.array([])
    for data in test_dataset:
        x, edge_index = data['x'], data['edge_index']
        z = model.encode(x, edge_index)
        neg_edge_index = negative_sampling(edge_index, z.size(0))
        auc, ap = model.test(z, edge_index, neg_edge_index)
        aucs = np.append(aucs, auc)
        aps = np.append(aps, ap)
    print('Test AUC: {:.3f}, AP: {:.3f}'.format(
        np.average(aucs), np.average(aps)))


def negative_sampling(edge_index, num_nodes):
    adj = [[False for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes):
        adj[i][i] = True
    for i in range(min(len(edge_index[0]), len(edge_index[1]))):
        adj[edge_index[0][i]][edge_index[1][i]] = True
    src = np.array([], dtype=np.int64)
    dst = np.array([], dtype=np.int64)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if not adj[i][j]:
                src = np.append(src, i)
                dst = np.append(dst, j)
    src = torch.from_numpy(src)
    dst = torch.from_numpy(dst)
    neg_edge_index = torch.stack((src, dst))
    return neg_edge_index


if __name__ == '__main__':
    main()
