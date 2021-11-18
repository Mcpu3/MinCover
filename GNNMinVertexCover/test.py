import os
from multiprocessing import Pool, freeze_support
import sys
from time import time

import dgl
from dgl.data import DGLDataset
from dgl.nn import GraphConv
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


def main():
    dataset = MinVertexCoverDataset()
    n_train = dataset.n_train
    gs = []
    labels = []
    for g in dataset[n_train:]:
        labels.append(g.ndata['labels'])
        g = dgl.remove_self_loop(g)
        g = dgl.to_networkx(g)
        g = nx.Graph(g)
        gs.append(g)
    acc, auc, ap, sum, avg = test_min_cover(gs, labels)
    print('Min Cover:')
    print('\tAcc: {:.3f}, AUC: {:.3f}, AP: {:.3f}, Sum: {:.3f}s, Avg: {:.3f}s'.format(
        acc, auc, ap, sum, avg))
    acc, auc, ap, sum, avg = test_approx_min_cover(gs, labels)
    print('Approx Min Cover:')
    print('\tAcc: {:.3f}, AUC: {:.3f}, AP: {:.3f}, Sum: {:.3f}s, Avg: {:.3f}s'.format(
        acc, auc, ap, sum, avg))
    model = GCN(dataset.n_feats, [16, 8, 4], dataset.n_classes)
    model.load_state_dict(torch.load('./models/model.pth'))
    model.eval()
    acc, auc, ap, sum, avg = test_test(dataset, model, labels)
    print('Test:')
    print('\tAcc: {:.3f}, AUC: {:.3f}, AP: {:.3f}, Sum: {:.3f}s, Avg: {:.3f}s'.format(
        acc, auc, ap, sum, avg))


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


def min_cover(g):
    min_cover = set()
    min_weight = sys.maxsize
    for i in range(2 ** g.number_of_nodes()):
        nodes = set()
        edges = set()
        for j in range(g.number_of_nodes()):
            if (i >> j) & 1:
                nodes.add(j)
                for k in g.adj[j]:
                    if j < k:
                        edges.add((j, k))
                    else:
                        edges.add((k, j))
        if edges == set(g.edges()):
            if len(nodes) < min_weight:
                min_cover = nodes
                min_weight = len(nodes)
    min_cover = np.array(list(min_cover))
    labels = np.array([0 for _ in range(g.number_of_nodes())])
    for node in min_cover:
        labels[node] = 1
    labels = torch.tensor(labels, dtype=torch.int64)
    return labels


def test_min_cover(gs, labels):
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    times_elapsed = np.array([])
    with tqdm(gs) as pbar:
        for g, label in zip(pbar, labels):
            time_start = time()
            pred = min_cover(g)
            time_end = time()
            acc = sklearn.metrics.accuracy_score(label, pred)
            accs = np.append(accs, acc)
            auc = sklearn.metrics.roc_auc_score(label, pred)
            aucs = np.append(aucs, auc)
            ap = sklearn.metrics.average_precision_score(label, pred)
            aps = np.append(aps, ap)
            time_elapsed = time_end - time_start
            times_elapsed = np.append(times_elapsed, time_elapsed)
            pbar.set_postfix_str('acc: {:.3f}, auc: {:.3f}, ap: {:.3f}, elapsed_time: {:.3f}s'.format(
                np.average(accs), np.average(aucs), np.average(aps), time_elapsed))
    return np.average(accs), np.average(aucs), np.average(aps), np.sum(times_elapsed), np.average(times_elapsed)


def approx_min_cover(g):
    min_cover = nx.algorithms.approximation.min_weighted_vertex_cover(g)
    min_cover = np.array(list(min_cover))
    labels = np.array([0 for _ in range(g.number_of_nodes())])
    for node in min_cover:
        labels[node] = 1
    labels = torch.tensor(labels, dtype=torch.int64)
    return labels


def test_approx_min_cover(gs, labels):
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    times_elapsed = np.array([])
    with tqdm(gs) as pbar:
        for g, label in zip(pbar, labels):
            time_start = time()
            pred = approx_min_cover(g)
            time_end = time()
            acc = sklearn.metrics.accuracy_score(label, pred)
            accs = np.append(accs, acc)
            auc = sklearn.metrics.roc_auc_score(label, pred)
            aucs = np.append(aucs, auc)
            ap = sklearn.metrics.average_precision_score(label, pred)
            aps = np.append(aps, ap)
            time_elapsed = time_end - time_start
            times_elapsed = np.append(times_elapsed, time_elapsed)
            pbar.set_postfix_str('acc: {:.3f}, auc: {:.3f}, ap: {:.3f}, elapsed_time: {:.3f}s'.format(
                np.average(accs), np.average(aucs), np.average(aps), time_elapsed))
    return np.average(accs), np.average(aucs), np.average(aps), np.sum(times_elapsed), np.average(times_elapsed)


def test(g, model):
    feats = g.ndata['feats']
    logits = model(g, feats)
    pred = logits.argmax(1)
    return pred


def test_test(dataset, model, labels):
    n_train = dataset.n_train
    test_dataset = dataset[n_train:]
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    times_elapsed = np.array([])
    with tqdm(test_dataset) as pbar:
        for g, label in zip(pbar, labels):
            time_start = time()
            pred = test(g, model)
            time_end = time()
            acc = sklearn.metrics.accuracy_score(label, pred)
            accs = np.append(accs, acc)
            auc = sklearn.metrics.roc_auc_score(label, pred)
            aucs = np.append(aucs, auc)
            ap = sklearn.metrics.average_precision_score(label, pred)
            aps = np.append(aps, ap)
            time_elapsed = time_end - time_start
            times_elapsed = np.append(times_elapsed, time_elapsed)
            pbar.set_postfix_str('acc: {:.3f}, auc: {:.3f}, ap: {:.3f}, elapsed_time: {:.3f}s'.format(
                np.average(accs), np.average(aucs), np.average(aps), time_elapsed))
    return np.average(accs), np.average(aucs), np.average(aps), np.sum(times_elapsed), np.average(times_elapsed)


if __name__ == '__main__':
    freeze_support()
    main()
