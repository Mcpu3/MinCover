from argparse import ArgumentParser
import copy
import os
from multiprocessing import Pool, freeze_support
from time import time

import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from dataset import Dataset, Graphs
from gcn import GCN
from min_vertex_cover import min_vertex_cover, min_vertex_cover_approx, min_vertex_cover_with_supervised_learning, min_vertex_cover_with_supervised_learning_1


def main(number_of_x, without_min_covers, without_approx_min_covers, without_with_supervised_learning, without_with_supervised_learning_1, path):
    dataset = Dataset(number_of_x, path)
    graphs = Graphs(path)
    dataset_test = dataset[dataset.number_of_train:]
    graphs_test = graphs[dataset.number_of_train:]
    model = None
    if (not without_with_supervised_learning) or (not without_with_supervised_learning_1):
        model = GCN(dataset.number_of_x, dataset.number_of_classes)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        model.eval()
    labels = []
    for data in dataset_test:
        labels.append(data.ndata['label'])
    test(dataset_test, model, graphs_test, labels, without_min_covers, without_approx_min_covers, without_with_supervised_learning, without_with_supervised_learning_1, path, dataset.number_of_train)


def test(dataset, model, graphs, labels, without_min_covers, without_approx_min_covers, without_with_supervised_learning, without_with_supervised_learning_1, path, number_of_train):
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        if not without_min_covers:
            accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers = test_min_vertex_cover(graphs, labels)
            for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers)):
                summary_writer.add_scalar('Acc/TestMinCover', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestMinCover', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestMinCover', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestMinCover', time_elapsed, number_of_train + index)
        if not without_approx_min_covers:
            accs_of_min_cover_approx, aucs_of_min_cover_approx, aps_of_min_cover_approx, times_elapsed_of_min_cover_approx = test_min_vertex_cover_approx(graphs, labels)
            for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_min_cover_approx, aucs_of_min_cover_approx, aps_of_min_cover_approx, times_elapsed_of_min_cover_approx)):
                summary_writer.add_scalar('Acc/TestApproxMinCover', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestApproxMinCover', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestApproxMinCover', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestApproxMinCover', time_elapsed, number_of_train + index)
        if not without_with_supervised_learning:
            accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning = test_with_supervised_learning(dataset, model, labels)
            for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning)):
                summary_writer.add_scalar('Acc/TestWithSupervisedLearning', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestWithSupervisedLearning', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestWithSupervisedLearning', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestWithSupervisedLearning', time_elapsed, number_of_train + index)
        if not without_with_supervised_learning_1:
            accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1 = test_with_supervised_learning_1(dataset, model, labels)
            for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1)):
                summary_writer.add_scalar('Acc/TestWithSupervisedLearning1', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestWithSupervisedLearning1', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestWithSupervisedLearning1', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestWithSupervisedLearning1', time_elapsed, number_of_train + index)


def test_min_vertex_cover(graphs, labels):
    accs, aucs, aps, times_elapsed = [], [], [], []
    min_covers_and_times_elapsed = process_map(min_vertex_cover_with_time_elapsed_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count() + 1)
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
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


def test_min_vertex_cover_approx(graphs, labels):
    accs, aucs, aps, times_elapsed = [], [], [], []
    min_covers_and_times_elapsed = []
    for graph in tqdm(graphs):
        min_covers_and_times_elapsed.append(min_vertex_cover_approx_with_time_elapsed((graph,)))
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
    return accs, aucs, aps, times_elapsed


def min_vertex_cover_approx_with_time_elapsed(arguments):
    graph = arguments[0]
    time_start = time()
    min_cover = min_vertex_cover_approx((graph,))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


def test_with_supervised_learning(dataset, model, labels):
    accs, aucs, aps, times_elapsed = [], [], [], []
    min_covers_and_times_elapsed = []
    for data in tqdm(dataset):
        min_covers_and_times_elapsed.append(with_supervised_learning_with_time_elapsed((data, model)))
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
    return accs, aucs, aps, times_elapsed


def with_supervised_learning_with_time_elapsed(arguments):
    dataset, model = arguments
    time_start = time()
    min_cover = min_vertex_cover_with_supervised_learning((dataset, model))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


def test_with_supervised_learning_1(dataset, model, labels):
    accs, aucs, aps, times_elapsed = [], [], [], []
    min_covers_and_times_elapsed = []
    for data in tqdm(dataset):
        min_covers_and_times_elapsed.append(with_supervised_learning_1_with_time_elapsed((data, model)))
    for (min_cover, time_elapsed), label, data in zip(min_covers_and_times_elapsed, labels, dataset):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
    return accs, aucs, aps, times_elapsed


def with_supervised_learning_1_with_time_elapsed(arguments):
    dataset, model = arguments
    time_start = time()
    min_cover = min_vertex_cover_with_supervised_learning_1((dataset, model))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_x', type=int, default=32)
    argument_parser.add_argument('--without_min_covers', action='store_true')
    argument_parser.add_argument('--without_approx_min_covers', action='store_true')
    argument_parser.add_argument('--without_with_supervised_learning', action='store_true')
    argument_parser.add_argument('--without_with_supervised_learning_1', action='store_true')
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_x = arguments.number_of_x
    without_min_covers = arguments.without_min_covers
    without_approx_min_covers = arguments.without_approx_min_covers
    without_with_supervised_learning = arguments.without_with_supervised_learning
    without_with_supervised_learning_1 = arguments.without_with_supervised_learning_1
    path = os.path.join('./runs/', arguments.path)
    main(number_of_x, without_min_covers, without_approx_min_covers, without_with_supervised_learning, without_with_supervised_learning_1, path)
