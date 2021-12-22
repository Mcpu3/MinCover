from argparse import ArgumentParser
from multiprocessing import Pool, freeze_support
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from min_vertex_cover import min_vertex_cover, min_vertex_cover_approx, min_vertex_cover_with_supervised_learning, min_vertex_cover_with_supervised_learning_1, min_vertex_cover_with_supervised_learning_2
from dataset import Dataset, Graphs
from gcn import GCN

def main(number_of_x, without_min_covers, without_min_covers_approx, without_with_supervised_learning, without_with_supervised_learning_1,  without_with_supervised_learning_2, path):
    dataset = Dataset(number_of_x, path)
    graphs = Graphs(path)
    dataset_test = dataset[dataset.number_of_train:]
    graphs_test = graphs[dataset.number_of_train:]
    model = None
    if (not without_with_supervised_learning) or (not without_with_supervised_learning_1) or (not without_with_supervised_learning_2):
        model = GCN(dataset.number_of_x, dataset.number_of_classes)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        model.eval()
    if not without_min_covers:
        min_covers = process_map(min_vertex_cover_wrapper, [(graph,) for graph in graphs_test], max_workers=os.cpu_count() + 1)
        process_map(save_figure_wrapper, [(graph, min_cover, os.path.join(path, 'figures/min_covers/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, min_cover) in enumerate(zip(graphs_test, min_covers))], max_workers=os.cpu_count() + 1)
    if not without_min_covers_approx:
        min_covers_approx = []
        for graph in tqdm(graphs_test):
            min_covers_approx.append(min_vertex_cover_approx((graph,)))
        process_map(save_figure_wrapper, [(graph, min_cover, os.path.join(path, 'figures/approx_min_covers/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, min_cover) in enumerate(zip(graphs_test, min_covers_approx))], max_workers=os.cpu_count() + 1)
    if not without_with_supervised_learning:
        with_supervised_learning = []
        for data in tqdm(dataset_test):
            with_supervised_learning.append(min_vertex_cover_with_supervised_learning((data, model)))
        process_map(save_figure_wrapper, [(graph, min_cover, os.path.join(path, 'figures/with_supervised_learning/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, min_cover) in enumerate(zip(graphs_test, with_supervised_learning))], max_workers=os.cpu_count() + 1)
    if not without_with_supervised_learning_1:
        with_supervised_learning_1 = []
        for data in tqdm(dataset_test):
            with_supervised_learning_1.append(min_vertex_cover_with_supervised_learning_1((data, model)))
        process_map(save_figure_wrapper, [(graph, min_cover, os.path.join(path, 'figures/with_supervised_learning_1/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, min_cover) in enumerate(zip(graphs_test, with_supervised_learning_1))], max_workers=os.cpu_count() + 1)
    if not without_with_supervised_learning_2:
        with_supervised_learning_2 = []
        for data in tqdm(dataset_test):
            with_supervised_learning_2.append(min_vertex_cover_with_supervised_learning_2((data, model)))
        process_map(save_figure_wrapper, [(graph, min_cover, os.path.join(path, 'figures/with_supervised_learning_2/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, min_cover) in enumerate(zip(graphs_test, with_supervised_learning_2))], max_workers=os.cpu_count() + 1)


def min_vertex_cover_wrapper(arguments):
    graph = arguments[0]
    with Pool(1) as pool:
        min_cover = pool.map(min_vertex_cover, [[graph]])[0]
    return min_cover


def save_figure_wrapper(arguments):
    graph, min_cover, path = arguments
    with Pool(1) as pool:
        pool.map(save_figure, [[graph, min_cover, path]])


def save_figure(arguments):
    graph, min_cover, path = arguments
    nodes_color = ['#333' for _ in range(graph.number_of_nodes())]
    edges_color = ['#000' for _ in range(graph.number_of_edges())]
    for node in range(graph.number_of_nodes()):
        if node in min_cover:
            nodes_color[node] = '#009b9f'
    for index, edge in enumerate(graph.edges()):
        if edge[0] not in min_cover and edge[1] not in min_cover:
            edges_color[index] = '#942343'
    nx.draw_networkx(graph, nx.circular_layout(graph), node_color=nodes_color, edge_color=edges_color, font_color='#fff')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


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
    without_min_covers_approx = arguments.without_approx_min_covers
    without_with_supervised_learning = arguments.without_with_supervised_learning
    without_with_supervised_learning_1 = arguments.without_with_supervised_learning_1
    without_with_supervised_learning_2 = arguments.without_with_supervised_learning_2
    path = os.path.join('./runs/', arguments.path)
    main(number_of_x, without_min_covers, without_min_covers_approx, without_with_supervised_learning, without_with_supervised_learning_1, without_with_supervised_learning_2, path)
