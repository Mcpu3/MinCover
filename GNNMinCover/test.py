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
from min_vertex_cover import min_vertex_cover, min_vertex_cover_approx, min_vertex_cover_with_supervised_learning, min_vertex_cover_with_supervised_learning_1, min_vertex_cover_with_supervised_learning_2


def main(number_of_x, without_min_covers, without_approx_min_covers, without_with_supervised_learning, without_with_supervised_learning_1, without_with_supervised_learning_2, path):
    dataset = Dataset(number_of_x, path)
    graphs = Graphs(path)
    dataset_test = dataset[dataset.number_of_train:]
    graphs_test = graphs[dataset.number_of_train:]
    model = None
    if (not without_with_supervised_learning) or (not without_with_supervised_learning_1) or (not without_with_supervised_learning_2):
        model = GCN(dataset.number_of_x, dataset.number_of_classes)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        model.eval()
    labels = []
    for data in dataset_test:
        labels.append(data.ndata['label'])
    test(dataset_test, model, graphs_test, labels, without_min_covers, without_approx_min_covers, without_with_supervised_learning, without_with_supervised_learning_1, without_with_supervised_learning_2, path, dataset.number_of_train)


def test(dataset, model, graphs, labels, without_min_covers, without_approx_min_covers, without_with_supervised_learning, without_with_supervised_learning_1, without_with_supervised_learning_2, path, number_of_train):
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        if not without_min_covers:
            accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers, evaluations_1_of_min_cover, evaluations_2_of_min_cover = test_min_vertex_cover(graphs, labels)
            for index, (acc, auc, ap, time_elapsed, evaluation_1, evaluation_2) in enumerate(zip(accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers, evaluations_1_of_min_cover, evaluations_2_of_min_cover)):
                summary_writer.add_scalar('Acc/TestMinCover', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestMinCover', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestMinCover', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestMinCover', time_elapsed, number_of_train + index)
                summary_writer.add_scalar('Evaluation1/TestMinCover', evaluation_1, number_of_train + index)
                summary_writer.add_scalar('Evaluation2/TestMinCover', evaluation_2, number_of_train + index)
        if not without_approx_min_covers:
            accs_of_min_cover_approx, aucs_of_min_cover_approx, aps_of_min_cover_approx, times_elapsed_of_min_cover_approx, evaluations_1_of_min_cover_approx, evaluations_2_of_min_cover_approx = test_min_vertex_cover_approx(graphs, labels)
            for index, (acc, auc, ap, time_elapsed, evaluation_1, evaluation_2) in enumerate(zip(accs_of_min_cover_approx, aucs_of_min_cover_approx, aps_of_min_cover_approx, times_elapsed_of_min_cover_approx, evaluations_1_of_min_cover_approx, evaluations_2_of_min_cover_approx)):
                summary_writer.add_scalar('Acc/TestApproxMinCover', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestApproxMinCover', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestApproxMinCover', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestApproxMinCover', time_elapsed, number_of_train + index)
                summary_writer.add_scalar('Evaluation1/TestApproxMinCover', evaluation_1, number_of_train + index)
                summary_writer.add_scalar('Evaluation2/TestApproxMinCover', evaluation_2, number_of_train + index)
        if not without_with_supervised_learning:
            accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning, evaluations_1_of_with_supervised_learning, evaluations_2_of_with_supervised_learning = test_with_supervised_learning(dataset, graphs, model, labels)
            for index, (acc, auc, ap, time_elapsed, evaluation_1, evaluation_2) in enumerate(zip(accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning, evaluations_1_of_with_supervised_learning, evaluations_2_of_with_supervised_learning)):
                summary_writer.add_scalar('Acc/TestWithSupervisedLearning', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestWithSupervisedLearning', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestWithSupervisedLearning', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestWithSupervisedLearning', time_elapsed, number_of_train + index)
                summary_writer.add_scalar('Evaluation1/TestWithSupervisedLearning', evaluation_1, number_of_train + index)
                summary_writer.add_scalar('Evaluation2/TestWithSupervisedLearning', evaluation_2, number_of_train + index)
        if not without_with_supervised_learning_1:
            accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1, evaluations_1_of_with_supervised_learning_1, evaluations_2_of_with_supervised_learning_1 = test_with_supervised_learning_1(dataset, graphs, model, labels)
            for index, (acc, auc, ap, time_elapsed, evaluation_1, evaluation_2) in enumerate(zip(accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1, evaluations_1_of_with_supervised_learning_1, evaluations_2_of_with_supervised_learning_1)):
                summary_writer.add_scalar('Acc/TestWithSupervisedLearning1', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestWithSupervisedLearning1', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestWithSupervisedLearning1', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestWithSupervisedLearning1', time_elapsed, number_of_train + index)
                summary_writer.add_scalar('Evaluation1/TestWithSupervisedLearning1', evaluation_1, number_of_train + index)
                summary_writer.add_scalar('Evaluation2/TestWithSupervisedLearning1', evaluation_2, number_of_train + index)
        if not without_with_supervised_learning_2:
            accs_of_with_supervised_learning_2, aucs_of_with_supervised_learning_2, aps_of_with_supervised_learning_2, times_elapsed_of_with_supervised_learning_2, evaluations_1_of_with_supervised_learning_2, evaluations_2_of_with_supervised_learning_2 = test_with_supervised_learning_2(dataset, graphs, model, labels)
            for index, (acc, auc, ap, time_elapsed, evaluation_1, evaluation_2) in enumerate(zip(accs_of_with_supervised_learning_2, aucs_of_with_supervised_learning_2, aps_of_with_supervised_learning_2, times_elapsed_of_with_supervised_learning_2, evaluations_1_of_with_supervised_learning_2, evaluations_2_of_with_supervised_learning_2)):
                summary_writer.add_scalar('Acc/TestWithSupervisedLearning2', acc, number_of_train + index)
                summary_writer.add_scalar('AUC/TestWithSupervisedLearning2', auc, number_of_train + index)
                summary_writer.add_scalar('AP/TestWithSupervisedLearning2', ap, number_of_train + index)
                summary_writer.add_scalar('ElapsedTime/TestWithSupervisedLearning2', time_elapsed, number_of_train + index)
                summary_writer.add_scalar('Evaluation1/TestWithSupervisedLearning2', evaluation_1, number_of_train + index)
                summary_writer.add_scalar('Evaluation2/TestWithSupervisedLearning2', evaluation_2, number_of_train + index)


def test_min_vertex_cover(graphs, labels):
    accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2 = [], [], [], [], [], []
    min_covers_and_times_elapsed = process_map(min_vertex_cover_with_time_elapsed_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count() + 1)
    for (min_cover, time_elapsed), graph, label in zip(min_covers_and_times_elapsed, graphs, labels):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        evaluation_1 = evaluate_1(label, min_cover)
        evaluation_2 = evaluate_2(min_cover, graph)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
        evaluations_1.append(evaluation_1)
        evaluations_2.append(evaluation_2)
    return accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2


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
    accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2 = [], [], [], [], [], []
    min_covers_and_times_elapsed = []
    for graph in tqdm(graphs):
        min_covers_and_times_elapsed.append(min_vertex_cover_approx_with_time_elapsed((graph,)))
    for (min_cover, time_elapsed), graph, label in zip(min_covers_and_times_elapsed, graphs, labels):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        evaluation_1 = evaluate_1(label, min_cover)
        evaluation_2 = evaluate_2(min_cover, graph)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
        evaluations_1.append(evaluation_1)
        evaluations_2.append(evaluation_2)
    return accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2


def min_vertex_cover_approx_with_time_elapsed(arguments):
    graph = arguments[0]
    time_start = time()
    min_cover = min_vertex_cover_approx((graph,))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


def test_with_supervised_learning(dataset, graphs, model, labels):
    accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2 = [], [], [], [], [], []
    min_covers_and_times_elapsed = []
    for data in tqdm(dataset):
        min_covers_and_times_elapsed.append(with_supervised_learning_with_time_elapsed((data, model)))
    for (min_cover, time_elapsed), graph, label in zip(min_covers_and_times_elapsed, graphs, labels):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        evaluation_1 = evaluate_1(label, min_cover)
        evaluation_2 = evaluate_2(min_cover, graph)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
        evaluations_1.append(evaluation_1)
        evaluations_2.append(evaluation_2)
    return accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2


def with_supervised_learning_with_time_elapsed(arguments):
    dataset, model = arguments
    time_start = time()
    min_cover = min_vertex_cover_with_supervised_learning((dataset, model))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


def test_with_supervised_learning_1(dataset, graphs, model, labels):
    accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2 = [], [], [], [], [], []
    min_covers_and_times_elapsed = []
    for data in tqdm(dataset):
        min_covers_and_times_elapsed.append(with_supervised_learning_1_with_time_elapsed((data, model)))
    for (min_cover, time_elapsed), data, graph, label in zip(min_covers_and_times_elapsed, dataset, graphs, labels):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        evaluation_1 = evaluate_1(label, min_cover)
        evaluation_2 = evaluate_2(min_cover, graph)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
        evaluations_1.append(evaluation_1)
        evaluations_2.append(evaluation_2)
    return accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2


def with_supervised_learning_1_with_time_elapsed(arguments):
    dataset, model = arguments
    time_start = time()
    min_cover = min_vertex_cover_with_supervised_learning_1((dataset, model))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


def test_with_supervised_learning_2(dataset, graphs, model, labels):
    accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2 = [], [], [], [], [], []
    min_covers_and_times_elapsed = []
    for data in tqdm(dataset):
        min_covers_and_times_elapsed.append(with_supervised_learning_2_with_time_elapsed((data, model)))
    for (min_cover, time_elapsed), data, graph, label in zip(min_covers_and_times_elapsed, dataset, graphs, labels):
        min_cover_copy = copy.deepcopy(min_cover)
        min_cover = [0 for _ in range(len(label))]
        for node in min_cover_copy:
            min_cover[node] = 1
        min_cover = torch.tensor(min_cover, dtype=torch.int64)
        acc = sklearn.metrics.accuracy_score(label, min_cover)
        auc = sklearn.metrics.roc_auc_score(label, min_cover)
        ap = sklearn.metrics.average_precision_score(label, min_cover)
        evaluation_1 = evaluate_1(label, min_cover)
        evaluation_2 = evaluate_2(min_cover, graph)
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        times_elapsed.append(time_elapsed)
        evaluations_1.append(evaluation_1)
        evaluations_2.append(evaluation_2)
    return accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2


def with_supervised_learning_2_with_time_elapsed(arguments):
    dataset, model = arguments
    time_start = time()
    min_cover = min_vertex_cover_with_supervised_learning_2((dataset, model))
    time_end = time()
    time_elapsed = time_end - time_start
    return min_cover, time_elapsed


# labelとmin_coverの濃度を評価
def evaluate_1(label, min_cover):
    return -(torch.count_nonzero(min_cover) - torch.count_nonzero(label)) / torch.count_nonzero(label) + 1.0


# 接続された頂点がmin_coverに無い辺を評価
def evaluate_2(min_cover, graph):
    min_cover_copy = copy.deepcopy(min_cover)
    min_cover = set()
    for node in range(len(min_cover_copy)):
        if min_cover_copy[node]:
            min_cover.add(node)
    edges = []
    for edge_source, edge_destination in graph.edges():
        if (edge_source not in min_cover) and (edge_destination not in min_cover):
            edges.append((edge_source, edge_destination))
    return 1.0 - len(edges) / graph.number_of_edges()


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_x', type=int, default=32)
    argument_parser.add_argument('--without_min_covers', action='store_true')
    argument_parser.add_argument('--without_approx_min_covers', action='store_true')
    argument_parser.add_argument('--without_with_supervised_learning', action='store_true')
    argument_parser.add_argument('--without_with_supervised_learning_1', action='store_true')
    argument_parser.add_argument('--without_with_supervised_learning_2', action='store_true')
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_x = arguments.number_of_x
    without_min_covers = arguments.without_min_covers
    without_approx_min_covers = arguments.without_approx_min_covers
    without_with_supervised_learning = arguments.without_with_supervised_learning
    without_with_supervised_learning_1 = arguments.without_with_supervised_learning_1
    without_with_supervised_learning_2 = arguments.without_with_supervised_learning_2
    path = os.path.join('./runs/', arguments.path)
    main(number_of_x, without_min_covers, without_approx_min_covers, without_with_supervised_learning, without_with_supervised_learning_1, without_with_supervised_learning_2, path)
