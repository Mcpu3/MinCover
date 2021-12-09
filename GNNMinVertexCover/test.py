from argparse import ArgumentParser
import os
from multiprocessing import Pool, freeze_support
import sys
from time import time

import networkx as nx
import numpy as np
import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.concurrent import process_map

from dataset import Dataset, Graphs
from gcn import GCN


def main(number_of_x, path):
    dataset = Dataset(number_of_x, path)
    graphs = Graphs(path)
    dataset_test = dataset[dataset.number_of_train:]
    graphs_test = graphs[dataset.number_of_train:]
    model = GCN(dataset.number_of_x, dataset.number_of_classes)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
    model.eval()
    labels = []
    for data in dataset_test:
        labels.append(data.ndata['label'])
    test(dataset_test, model, graphs_test, labels, path, dataset.number_of_train)


def test(dataset, model, graphs, labels, path, number_of_train):
    labels = []
    for data in dataset:
        labels.append(data.ndata['label'])
    accs_of_min_cover, aucs_of_min_cover, aps_of_min_cover, times_elapsed_of_min_cover = test_min_vertex_cover(graphs, labels)
    accs_of_min_cover_approx, aucs_of_min_cover_approx, aps_of_min_cover_approx, times_elapsed_of_min_cover_approx = test_min_vertex_cover_approx(graphs, labels)
    accs_of_min_cover_with_supervised_learning, aucs_of_min_cover_with_supervised_learning, aps_of_min_cover_with_supervised_learning, times_elapsed_of_min_cover_with_supervised_learning = test_min_vertex_cover_with_supervised_learning(dataset, model, labels)
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_min_cover, aucs_of_min_cover, aps_of_min_cover, times_elapsed_of_min_cover)):
            summary_writer.add_scalar('Acc/TestMinCover', acc, number_of_train + index)
            summary_writer.add_scalar('AUC/TestMinCover', auc, number_of_train + index)
            summary_writer.add_scalar('AP/TestMinCover', ap, number_of_train + index)
            summary_writer.add_scalar('ElapsedTime/TestMinCover', time_elapsed, number_of_train + index)
        for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_min_cover_approx, aucs_of_min_cover_approx, aps_of_min_cover_approx, times_elapsed_of_min_cover_approx)):
            summary_writer.add_scalar('Acc/TestApproxMinCover', acc, number_of_train + index)
            summary_writer.add_scalar('AUC/TestApproxMinCover', auc, number_of_train + index)
            summary_writer.add_scalar('AP/TestApproxMinCover', ap, number_of_train + index)
            summary_writer.add_scalar('ElapsedTime/TestApproxMinCover', time_elapsed, number_of_train + index)
        for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_min_cover_with_supervised_learning, aucs_of_min_cover_with_supervised_learning, aps_of_min_cover_with_supervised_learning, times_elapsed_of_min_cover_with_supervised_learning)):
            summary_writer.add_scalar('Acc/TestMinCoverWithSupervisedLearning', acc, number_of_train + index)
            summary_writer.add_scalar('AUC/TestMinCoverWithSupervisedLearning', auc, number_of_train + index)
            summary_writer.add_scalar('AP/TestMinCoverWithSupervisedLearning', ap, number_of_train + index)
            summary_writer.add_scalar('ElapsedTime/TestMinCoverWithSupervisedLearning', time_elapsed, number_of_train + index)


def test_min_vertex_cover(graphs, labels):
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    times_elapsed = np.array([])
    min_covers_and_times_elapsed = process_map(min_vertex_cover_with_time_elapsed_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count() + 1)
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = min_cover
        min_cover = np.array([0 for _ in range(len(label))])
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        accs = np.append(accs, acc)
        aucs = np.append(aucs, auc)
        aps = np.append(aps, ap)
        times_elapsed = np.append(times_elapsed, time_elapsed)
    return accs, aucs, aps, times_elapsed


def min_vertex_cover_with_time_elapsed_wrapper(arguments):
    graph = arguments[0]
    with Pool(1) as pool:
        min_cover, time_elapsed = pool.map(min_vertex_cover_with_time_elapsed, [[graph]])[0]
    return min_cover, time_elapsed


def min_vertex_cover_with_time_elapsed(arguments):
    graph = arguments[0]
    time_start = time()
    min_cover = min_vertex_cover((graph,))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


def min_vertex_cover(arguments):
    graph = arguments[0]
    min_cover = set()
    min_weight = sys.maxsize
    for i in range(2 ** graph.number_of_nodes()):
        nodes = set()
        edges = set()
        for j in range(graph.number_of_nodes()):
            if (i >> j) & 1:
                nodes.add(j)
                for k in graph.adj[j]:
                    if j < k:
                        edges.add((j, k))
                    else:
                        edges.add((k, j))
        if edges == set(graph.edges()):
            if len(nodes) < min_weight:
                min_cover = nodes
                min_weight = len(nodes)
    return min_cover


def test_min_vertex_cover_approx(graphs, labels):
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    times_elapsed = np.array([])
    min_covers_and_times_elapsed = process_map(min_vertex_cover_approx_with_time_elapsed_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count() + 1)
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = min_cover
        min_cover = np.array([0 for _ in range(len(label))])
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        accs = np.append(accs, acc)
        aucs = np.append(aucs, auc)
        aps = np.append(aps, ap)
        times_elapsed = np.append(times_elapsed, time_elapsed)
    return accs, aucs, aps, times_elapsed


def min_vertex_cover_approx_with_time_elapsed_wrapper(arguments):
    graph = arguments[0]
    with Pool(1) as pool:
        min_cover, time_elapsed = pool.map(min_vertex_cover_approx_with_time_elapsed, [[graph]])[0]
    return min_cover, time_elapsed


def min_vertex_cover_approx_with_time_elapsed(arguments):
    graph = arguments[0]
    time_start = time()
    min_cover = min_vertex_cover_approx((graph,))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


def min_vertex_cover_approx(arguments):
    graph = arguments[0]
    min_cover = nx.algorithms.approximation.min_weighted_vertex_cover(graph)
    return min_cover


def test_min_vertex_cover_with_supervised_learning(dataset, model, labels):
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    times_elapsed = np.array([])
    min_covers_and_times_elapsed = process_map(min_vertex_cover_with_supervised_learning_with_time_elapsed_wrapper, [(graph, model) for graph in dataset], max_workers=os.cpu_count() + 1)
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = min_cover
        min_cover = np.array([0 for _ in range(len(label))])
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        accs = np.append(accs, acc)
        aucs = np.append(aucs, auc)
        aps = np.append(aps, ap)
        times_elapsed = np.append(times_elapsed, time_elapsed)
    return accs, aucs, aps, times_elapsed


def min_vertex_cover_with_supervised_learning_with_time_elapsed_wrapper(arguments):
    dataset, model = arguments
    with Pool(1) as pool:
        min_cover, time_elapsed = pool.map(min_vertex_cover_with_supervised_learning_with_time_elapsed, [[dataset, model]])[0]
    return min_cover, time_elapsed


def min_vertex_cover_with_supervised_learning_with_time_elapsed(arguments):
    dataset, model = arguments
    time_start = time()
    min_cover = min_vertex_cover_with_supervised_learning((dataset, model))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


def min_vertex_cover_with_supervised_learning(arguments):
    dataset, model = arguments
    x = dataset.ndata['x']
    min_cover_old = model(dataset, x).argmax(1)
    min_cover = set()
    for node in range(len(min_cover_old)):
        if min_cover_old[node]:
            min_cover.add(node)
    return min_cover


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_x', type=int, default=32)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_x = arguments.number_of_x
    path = os.path.join('./runs/', arguments.path)
    main(number_of_x, path)
