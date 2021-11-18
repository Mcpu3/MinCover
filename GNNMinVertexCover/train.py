import os
from multiprocessing import Pool, freeze_support
import dgl
from dgl.data import DGLDataset
from dgl.nn import GraphConv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.metrics
from tqdm.contrib.concurrent import process_map
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
    dataset = MinVertexCoverDataset()
    model = GCN(dataset.n_feats, [16, 8, 4], dataset.n_classes)
    model.train()
    model = train(dataset, model)
    torch.save(model.state_dict(), './models/model.pth')
    model.eval()
    test(dataset, model)


class MinVertexCoverDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='min_vertex_cover')

    def process(self):
        self.n_classes = 2
        self.n_feats = 32
        self.gs = []
        nodes = pd.read_csv('./dataset/nodes.csv')
        edges = pd.read_csv('./dataset/edges.csv')
        nodes_group = nodes.groupby('g_id')
        edges_group = edges.groupby('g_id')
        n_nodes_dict = {}
        labels_dict = {}
        for g_id in nodes_group.groups:
            nodes_of_id = nodes_group.get_group(g_id)
            n_nodes = nodes_of_id.to_numpy().shape[0]
            labels = torch.from_numpy(nodes_of_id['labels'].to_numpy())
            n_nodes_dict[g_id] = n_nodes
            labels_dict[g_id] = labels
        for g_id in nodes_group.groups:
            src = torch.empty(0, dtype=torch.int64)
            dst = torch.empty(0, dtype=torch.int64)
            if g_id in edges_group.groups.keys():
                edges_of_id = edges_group.get_group(g_id)
                src = torch.from_numpy(edges_of_id['src'].to_numpy())
                dst = torch.from_numpy(edges_of_id['dst'].to_numpy())
            n_nodes = n_nodes_dict[g_id]
            labels = labels_dict[g_id]
            g = dgl.graph((src, dst), num_nodes=n_nodes)
            g = dgl.add_self_loop(g)
            feats = torch.rand([n_nodes, self.n_feats], dtype=torch.float32)
            g.ndata['feats'] = feats
            g.ndata['labels'] = labels
            self.gs.append(g)
        n_gs = len(self.gs)
        n_train = int(n_gs * 0.8)
        self.n_train = n_train

    def __getitem__(self, i):
        return self.gs[i]

    def __len__(self):
        return len(self.gs)


class GCN(nn.Module):
    def __init__(self, n_feats, n_hidden_feats, n_classes):
        super(GCN, self).__init__()
        self.n_hiddens = [2, 2, 2]
        self.hiddens = []
        self.biases = []
        self.hiddens.append(GraphConv(n_feats, n_hidden_feats[0]))
        self.biases.append(nn.Parameter(torch.zeros(1)))
        for hidden_ind, n_hidden in enumerate(self.n_hiddens):
            for _ in range(n_hidden):
                self.hiddens.append(
                    GraphConv(n_hidden_feats[hidden_ind], n_hidden_feats[hidden_ind]))
                self.biases.append(nn.Parameter(torch.zeros(1)))
            if hidden_ind < len(self.n_hiddens) - 1:
                self.hiddens.append(
                    GraphConv(n_hidden_feats[hidden_ind], n_hidden_feats[hidden_ind + 1]))
                self.biases.append(nn.Parameter(torch.zeros(1)))
        self.hiddens.append(GraphConv(n_hidden_feats[-1], n_classes))
        self.biases.append(nn.Parameter(torch.zeros(1)))
        self.hiddens = nn.ModuleList(self.hiddens)
        self.biases = nn.ParameterList(self.biases)

    def forward(self, g, feats):
        h = feats
        for hidden_ind, hidden in enumerate(self.hiddens):
            h = hidden(g, h)
            h = F.relu(h + self.biases[hidden_ind])
        h = F.softmax(h, dim=1)
        return h


def train(dataset, model):
    n_train = dataset.n_train
    train_dataset = dataset[:n_train]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with SummaryWriter() as writer:
        for e in tqdm(range(64)):
            accs = np.array([])
            aucs = np.array([])
            aps = np.array([])
            with tqdm(train_dataset) as pbar:
                for g in pbar:
                    feats = g.ndata['feats']
                    labels = g.ndata['labels']
                    logits = model(g, feats)
                    pred = logits.argmax(1)
                    acc = sklearn.metrics.accuracy_score(labels, pred)
                    accs = np.append(accs, acc)
                    auc = sklearn.metrics.roc_auc_score(labels, pred)
                    aucs = np.append(aucs, auc)
                    ap = sklearn.metrics.average_precision_score(labels, pred)
                    aps = np.append(aps, ap)
                    loss = F.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix_str(
                        'epoch: {}, loss: {:.3f}, acc: {:.3f}, auc: {:.3f}, ap: {:.3f}'.format(e, loss, np.average(accs), np.average(aucs), np.average(aps)))
            writer.add_scalar('Loss/train', loss, e)
            writer.add_scalar('Acc/train', np.average(accs), e)
            writer.add_scalar('AUC/train', np.average(aucs), e)
            writer.add_scalar('AP/train', np.average(aps), e)

    return model


def test(dataset, model):
    n_train = dataset.n_train
    test_dataset = dataset[n_train:]
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    gs = []
    min_covers = []
    with tqdm(test_dataset) as pbar:
        for g in pbar:
            feats = g.ndata['feats']
            labels = g.ndata['labels']
            logits = model(g, feats)
            pred = logits.argmax(1)
            acc = sklearn.metrics.accuracy_score(labels, pred)
            accs = np.append(accs, acc)
            auc = sklearn.metrics.roc_auc_score(labels, pred)
            aucs = np.append(aucs, auc)
            ap = sklearn.metrics.average_precision_score(labels, pred)
            aps = np.append(aps, ap)
            loss = F.cross_entropy(logits, labels)
            g = dgl.remove_self_loop(g)
            g = dgl.DGLGraph.to_networkx(g)
            g = nx.Graph(g)
            min_cover = set()
            for node in range(g.number_of_nodes()):
                if pred[node]:
                    min_cover.add(node)
            gs.append(g)
            min_covers.append(min_cover)
            pbar.set_postfix_str(
                'loss: {:.3f}, acc: {:.3f}, auc: {:.3f}, ap: {:.3f}'.format(loss, np.average(accs), np.average(aucs), np.average(aps)))
    print('Loss: {:.3f}, Acc: {:.3f}, AUC: {:.3f}, AP: {:.3f}'.format(
        loss, np.average(accs), np.average(aucs), np.average(aps)))
    process_map(savefig_min_cover, [(g, min_cover, './fig/tests/{}.jpg'.format(n_train + g_ind))
                for g, min_cover, g_ind in zip(gs, min_covers, range(len(test_dataset)))], max_workers=os.cpu_count()+1)


def savefig_min_cover_wrapper(args):
    g, min_cover, path = args
    with Pool(1) as p:
        p.map(savefig_min_cover, [[g, min_cover, path]])


def savefig_min_cover(args):
    g, min_cover, path = args
    pos = nx.circular_layout(g)
    node_color = ['#333333'] * g.number_of_nodes()
    for node in min_cover:
        node_color[node] = '#009b9f'
    edge_color = ['#000000'] * g.number_of_edges()
    for edge_ind, edge in enumerate(g.edges()):
        if edge[0] not in min_cover and edge[1] not in min_cover:
            edge_color[edge_ind] = '#942343'
    nx.draw_networkx(g, pos, node_color=node_color,
                     edge_color=edge_color, font_color='#ffffff')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


if __name__ == '__main__':
    freeze_support()
    main()
