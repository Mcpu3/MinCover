from argparse import ArgumentParser
import os
from multiprocessing import Pool, freeze_support
import sys
from time import time

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.concurrent import process_map

from GNNMinVertexCover.dataset import Dataset
from gcn import GCN


def main(number_of_features, path):
    dataset = Dataset(number_of_features, path)
    test_dataset = dataset[dataset.number_of_train:]
    graphs = []
    labels = []
    for graph in test_dataset:
        labels.append(graph.ndata['labels'])
        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_networkx(graph)
        graph = nx.Graph(graph)
        graphs.append(graph)
    acc, auc, ap, time_elapsed = test_min_cover(graphs, labels, path, dataset.number_of_train)
    print('Min Cover:')
    print('\tAcc: {:.3f}, AUC: {:.3f}, AP: {:.3f}, ElapsedTime: {:.3f}s'.format(acc, auc, ap, time_elapsed))
    acc, auc, ap, time_elapsed = test_approx_min_cover(graphs, labels, path, dataset.number_of_train)
    print('Approx Min Cover:')
    print('\tAcc: {:.3f}, AUC: {:.3f}, AP: {:.3f}, ElapsedTime: {:.3f}s'.format(acc, auc, ap, time_elapsed))
    model = GCN(dataset.number_of_features, dataset.number_of_classes)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
    model.eval()
    acc, auc, ap, time_elapsed = test_test(test_dataset, model, labels, path, dataset.number_of_train)
    print('Test:')
    print('\tAcc: {:.3f}, AUC: {:.3f}, AP: {:.3f}, ElapsedTime: {:.3f}s'.format(acc, auc, ap, time_elapsed))
    predicts_and_times_elapsed = process_map(min_cover_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count()+1)
    predicts = []
    for predict, _ in predicts_and_times_elapsed:
        predicts.append(predict)
    process_map(savefigure_min_cover_wrapper, [(graph, predict, os.path.join(path, 'figures/min_covers/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, predict) in enumerate(zip(graphs, predicts))], max_workers=os.cpu_count()+1)
    predicts_and_times_elapsed = process_map(approx_min_cover_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count()+1)
    predicts = []
    for predict, _ in predicts_and_times_elapsed:
        predicts.append(predict)
    process_map(savefigure_min_cover_wrapper, [(graph, predict, os.path.join(path, 'figures/approx_min_covers/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, predict) in enumerate(zip(graphs, predicts))], max_workers=os.cpu_count()+1)
    predicts_and_times_elapsed = process_map(test_wrapper, [(graph, model) for graph in test_dataset], max_workers=os.cpu_count()+1)
    predicts = []
    for predict, _ in predicts_and_times_elapsed:
        predicts.append(predict)
    process_map(savefigure_min_cover_wrapper, [(graph, predict, os.path.join(path, 'figures/tests/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, predict) in enumerate(zip(graphs, predicts))], max_workers=os.cpu_count()+1)


def min_cover_wrapper(arguments):
    graph = arguments[0]
    with Pool(1) as pool:
        predict, time_elapsed = pool.map(min_cover, [[graph]])[0]
    return predict, time_elapsed


def min_cover(arguments):
    graph = arguments[0]
    time_start = time()
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
    min_cover = np.array(list(min_cover))
    predict = np.array([0 for _ in range(graph.number_of_nodes())])
    for node in min_cover:
        predict[node] = 1
    predict = torch.tensor(predict, dtype=torch.int64)
    time_end = time()
    time_elapsed = time_end - time_start
    return predict, time_elapsed


def test_min_cover(graphs, labels, path, number_of_train):
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    predicts_and_times_elapsed = process_map(min_cover_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count()+1)
    predicts = []
    times_elapsed = np.array([])
    for predict, time_elapsed in predicts_and_times_elapsed:
        predicts.append(predict)
        times_elapsed = np.append(times_elapsed, time_elapsed)
    for predict, label in zip(predicts, labels):
        acc = sklearn.metrics.accuracy_score(label, predict)
        accs = np.append(accs, acc)
        auc = sklearn.metrics.roc_auc_score(label, predict)
        aucs = np.append(aucs, auc)
        ap = sklearn.metrics.average_precision_score(label, predict)
        aps = np.append(aps, ap)
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs, aucs, aps, times_elapsed)):
            summary_writer.add_scalar('Acc/test', acc, number_of_train + index)
            summary_writer.add_scalar('AUC/test', auc, number_of_train + index)
            summary_writer.add_scalar('AP/test', ap, number_of_train + index)
            summary_writer.add_scalar('ElapsedTime/test', time_elapsed, number_of_train + index)
    return np.mean(accs), np.mean(aucs), np.mean(aps), np.mean(times_elapsed)


def approx_min_cover_wrapper(arguments):
    graph = arguments[0]
    with Pool(1) as pool:
        predict, time_elapsed = pool.map(approx_min_cover, [[graph]])[0]
    return predict, time_elapsed


def approx_min_cover(arguments):
    graph = arguments[0]
    time_start = time()
    min_cover = nx.algorithms.approximation.min_weighted_vertex_cover(graph)
    min_cover = np.array(list(min_cover))
    predict = np.array([0 for _ in range(graph.number_of_nodes())])
    for node in min_cover:
        predict[node] = 1
    predict = torch.tensor(predict, dtype=torch.int64)
    time_end = time()
    time_elapsed = time_end - time_start
    return predict, time_elapsed


def test_approx_min_cover(graphs, labels, path, number_of_train):
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    predicts_and_times_elapsed = process_map(approx_min_cover_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count()+1)
    predicts = []
    times_elapsed = np.array([])
    for predict, time_elapsed in predicts_and_times_elapsed:
        predicts.append(predict)
        times_elapsed = np.append(times_elapsed, time_elapsed)
    for predict, label in zip(predicts, labels):
        acc = sklearn.metrics.accuracy_score(label, predict)
        accs = np.append(accs, acc)
        auc = sklearn.metrics.roc_auc_score(label, predict)
        aucs = np.append(aucs, auc)
        ap = sklearn.metrics.average_precision_score(label, predict)
        aps = np.append(aps, ap)
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs, aucs, aps, times_elapsed)):
            summary_writer.add_scalar('Acc/test', acc, number_of_train + index)
            summary_writer.add_scalar('AUC/test', auc, number_of_train + index)
            summary_writer.add_scalar('AP/test', ap, number_of_train + index)
            summary_writer.add_scalar('ElapsedTime/test', time_elapsed, number_of_train + index)
    return np.mean(accs), np.mean(aucs), np.mean(aps), np.mean(times_elapsed)


def test_wrapper(arguments):
    graph, model = arguments
    with Pool(1) as pool:
        predict, time_elapsed = pool.map(test, [[graph, model]])[0]
    return predict, time_elapsed


def test(arguments):
    graph, model = arguments
    time_start = time()
    features = graph.ndata['features']
    predict = model(graph, features).argmax(1)
    time_end = time()
    time_elapsed = time_end - time_start
    return predict, time_elapsed


def test_test(dataset, model, labels, path, number_of_train):
    accs = np.array([])
    aucs = np.array([])
    aps = np.array([])
    predicts_and_times_elapsed = process_map(test_wrapper, [(graph, model) for graph in dataset], max_workers=os.cpu_count()+1)
    predicts = []
    times_elapsed = np.array([])
    for predict, time_elapsed in predicts_and_times_elapsed:
        predicts.append(predict)
        times_elapsed = np.append(times_elapsed, time_elapsed)
    for predict, label in zip(predicts, labels):
        acc = sklearn.metrics.accuracy_score(label, predict)
        accs = np.append(accs, acc)
        auc = sklearn.metrics.roc_auc_score(label, predict)
        aucs = np.append(aucs, auc)
        ap = sklearn.metrics.average_precision_score(label, predict)
        aps = np.append(aps, ap)
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs, aucs, aps, times_elapsed)):
            summary_writer.add_scalar('Acc/test', acc, number_of_train + index)
            summary_writer.add_scalar('AUC/test', auc, number_of_train + index)
            summary_writer.add_scalar('AP/test', ap, number_of_train + index)
            summary_writer.add_scalar('ElapsedTime/test', time_elapsed, number_of_train + index)
    return np.mean(accs), np.mean(aucs), np.mean(aps), np.mean(times_elapsed)


def savefigure_min_cover_wrapper(arguments):
    graph, min_cover, path = arguments
    with Pool(1) as pool:
        pool.map(savefigure_min_cover, [[graph, min_cover, path]])


def savefigure_min_cover(arguments):
    graph, min_cover, path = arguments
    position = nx.circular_layout(graph)
    nodes_color = ['#333333'] * graph.number_of_nodes()
    for node in range(graph.number_of_nodes()):
        if min_cover[node]:
            nodes_color[node] = '#009b9f'
    nx.draw_networkx(graph, position, node_color=nodes_color,
                     font_color='#ffffff')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_features', type=int, default=32)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_features = arguments.number_of_features
    path = os.path.join('./runs/', arguments.path)
    main(number_of_features, path)
