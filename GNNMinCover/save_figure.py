from argparse import ArgumentParser
from multiprocessing import Pool, freeze_support
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
from tqdm.contrib.concurrent import process_map

from min_vertex_cover import min_vertex_cover, min_vertex_cover_approx, min_vertex_cover_with_supervised_learning
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
    min_covers = process_map(min_vertex_cover_wrapper, [(graph,) for graph in graphs_test], max_workers=os.cpu_count() + 1)
    min_covers_approx = process_map(min_vertex_cover_approx_wrapper, [(graph,) for graph in graphs_test], max_workers=os.cpu_count() + 1)
    min_covers_with_supervised_learning = process_map(min_vertex_cover_with_supervised_learning_wrapper, [(data, model) for data in dataset_test], max_workers=os.cpu_count() + 1)
    process_map(save_figure_wrapper, [(graph, min_cover, os.path.join(path, 'figures/min_covers/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, min_cover) in enumerate(zip(graphs_test, min_covers))], max_workers=os.cpu_count() + 1)
    process_map(save_figure_wrapper, [(graph, min_cover, os.path.join(path, 'figures/approx_min_covers/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, min_cover) in enumerate(zip(graphs_test, min_covers_approx))], max_workers=os.cpu_count() + 1)
    process_map(save_figure_wrapper, [(graph, min_cover, os.path.join(path, 'figures/min_covers_with_supervised_learning/{}.jpg'.format(dataset.number_of_train + index))) for index, (graph, min_cover) in enumerate(zip(graphs_test, min_covers_with_supervised_learning))], max_workers=os.cpu_count() + 1)


def min_vertex_cover_wrapper(arguments):
    graph = arguments[0]
    with Pool(1) as pool:
        min_cover = pool.map(min_vertex_cover, [[graph]])[0]
    return min_cover


def min_vertex_cover_approx_wrapper(arguments):
    graph = arguments[0]
    with Pool(1) as pool:
        min_cover = pool.map(min_vertex_cover_approx, [[graph]])[0]
    return min_cover


def min_vertex_cover_with_supervised_learning_wrapper(arguments):
    data, model = arguments
    with Pool(1) as pool:
        min_cover = pool.map(min_vertex_cover_with_supervised_learning, [[data, model]])[0]
    return min_cover


def save_figure_wrapper(arguments):
    graph, min_cover, path = arguments
    with Pool(1) as pool:
        pool.map(save_figure, [[graph, min_cover, path]])


def save_figure(arguments):
    graph, min_cover, path = arguments
    nodes_color = ['#333' for _ in range(graph.number_of_nodes())]
    for node in range(graph.number_of_nodes()):
        if node in min_cover:
            nodes_color[node] = '#009b9f'
    nx.draw_networkx(graph, nx.circular_layout(graph), node_color=nodes_color, font_color='#fff')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_x', type=int, default=32)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_x = arguments.number_of_x
    path = os.path.join('./runs/', arguments.path)
    main(number_of_x, path)