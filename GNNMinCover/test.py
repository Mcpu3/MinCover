from argparse import ArgumentParser
import os
from multiprocessing import Pool, freeze_support
from time import time

import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.concurrent import process_map

from dataset import Dataset, Graphs
from gcn import GCN
from min_vertex_cover import min_vertex_cover, min_vertex_cover_approx, min_vertex_cover_with_supervised_learning


def main(number_of_x, without_min_covers, without_approx_min_covers, without_min_covers_with_supervised_learning, path):
    dataset = Dataset(number_of_x, path)
    graphs = Graphs(path)
    dataset_test = dataset[dataset.number_of_train:]
    graphs_test = graphs[dataset.number_of_train:]
    model = None
    if not without_min_covers_with_supervised_learning:
        model = GCN(dataset.number_of_x, dataset.number_of_classes)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        model.eval()
    labels = []
    for data in dataset_test:
        labels.append(data.ndata['label'])
    test(dataset_test, model, graphs_test, labels, without_min_covers, without_approx_min_covers, without_min_covers_with_supervised_learning, path, dataset.number_of_train)


def test(dataset, model, graphs, labels, without_min_covers, without_approx_min_covers, without_min_covers_with_supervised_learning, path, number_of_train):
    if not without_min_covers:
        accs_of_min_cover, aucs_of_min_cover, aps_of_min_cover, times_elapsed_of_min_cover = test_min_vertex_cover(graphs, labels)
    if not without_approx_min_covers:
        accs_of_min_cover_approx, aucs_of_min_cover_approx, aps_of_min_cover_approx, times_elapsed_of_min_cover_approx = test_min_vertex_cover_approx(graphs, labels)
    if not without_min_covers_with_supervised_learning:
        accs_of_min_cover_with_supervised_learning, aucs_of_min_cover_with_supervised_learning, aps_of_min_cover_with_supervised_learning, times_elapsed_of_min_cover_with_supervised_learning = test_min_vertex_cover_with_supervised_learning(dataset, model, labels)
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        if not without_min_covers:
            for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_min_cover, aucs_of_min_cover, aps_of_min_cover, times_elapsed_of_min_cover)):
                summary_writer.add_scalar('Acc/TestMinCover', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestMinCover', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestMinCover', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestMinCover', time_elapsed, number_of_train + index)
        if not without_approx_min_covers:
            for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_min_cover_approx, aucs_of_min_cover_approx, aps_of_min_cover_approx, times_elapsed_of_min_cover_approx)):
                summary_writer.add_scalar('Acc/TestApproxMinCover', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestApproxMinCover', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestApproxMinCover', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestApproxMinCover', time_elapsed, number_of_train + index)
        if not without_min_covers_with_supervised_learning:
            for index, (acc, auc, ap, time_elapsed) in enumerate(zip(accs_of_min_cover_with_supervised_learning, aucs_of_min_cover_with_supervised_learning, aps_of_min_cover_with_supervised_learning, times_elapsed_of_min_cover_with_supervised_learning)):
                summary_writer.add_scalar('Acc/TestMinCoverWithSupervisedLearning', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestMinCoverWithSupervisedLearning', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestMinCoverWithSupervisedLearning', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestMinCoverWithSupervisedLearning', time_elapsed, number_of_train + index)


def test_min_vertex_cover(graphs, labels):
    accs, aucs, aps, times_elapsed = [], [], [], []
    min_covers_and_times_elapsed = process_map(min_vertex_cover_with_time_elapsed_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count() + 1)
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = min_cover
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
    min_covers_and_times_elapsed = process_map(min_vertex_cover_approx_with_time_elapsed_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count() + 1)
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = min_cover
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


def test_min_vertex_cover_with_supervised_learning(dataset, model, labels):
    accs, aucs, aps, times_elapsed = [], [], [], []
    min_covers_and_times_elapsed = process_map(min_vertex_cover_with_supervised_learning_with_time_elapsed_wrapper, [(graph, model) for graph in dataset], max_workers=os.cpu_count() + 1)
    for (min_cover, time_elapsed), label in zip(min_covers_and_times_elapsed, labels):
        min_cover_copy = min_cover
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


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_x', type=int, default=32)
    argument_parser.add_argument('--without_min_covers', action='store_true')
    argument_parser.add_argument('--without_approx_min_covers', action='store_true')
    argument_parser.add_argument('--without_min_covers_with_supervised_learning', action='store_true')
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_x = arguments.number_of_x
    without_min_covers = arguments.without_min_covers
    without_approx_min_covers = arguments.without_approx_min_covers
    without_min_covers_with_supervised_learning = arguments.without_min_covers_with_supervised_learning
    path = os.path.join('./runs/', arguments.path)
    main(number_of_x, without_min_covers, without_approx_min_covers, without_min_covers_with_supervised_learning, path)
