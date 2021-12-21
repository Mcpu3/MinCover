from argparse import ArgumentParser
import copy
from multiprocessing import Pool, freeze_support
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import VGAE
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from dataset import Dataset, Graphs
from encoder import Encoder


def main(number_of_x, number_of_classes, path):
    dataset = Dataset(number_of_x, path)
    graphs = Graphs(path)
    model = VGAE(Encoder(number_of_x, number_of_classes))
    model.eval()
    process_map(save_figure_wrapper, [(graph, os.path.join(path, 'figures/A/{}.jpg'.format(index))) for index, graph in enumerate(graphs)], max_workers=os.cpu_count() + 1)
    for index, data in enumerate(tqdm(dataset)):
        edge_index, x = data['edge_index'], data['x']
        z = model.encode(x, edge_index)
        adjacency = model.decoder.forward_all(z)
        adjacency_copy = copy.deepcopy(adjacency)
        adjacency = [[0 for _ in range(len(adjacency_copy[0]))] for _ in range(len(adjacency_copy))]
        for i in range(len(adjacency)):
            for j in range(len(adjacency[i])):
                if adjacency_copy[i][j] >= 0.5:
                    adjacency[i][j] = 1
        graph = nx.from_numpy_matrix(np.array(adjacency))
        save_figure((graph, os.path.join(path, 'figures/A Tilda/{}.jpg'.format(index))))


def save_figure_wrapper(arguments):
    graph, path = arguments
    with Pool(1) as pool:
        pool.map(save_figure, [[graph, path]])


def save_figure(arguments):
    graph, path = arguments
    nodes_color = ['#333' for _ in range(graph.number_of_nodes())]
    nx.draw_networkx(graph, nx.circular_layout(graph), node_color=nodes_color, font_color='#fff')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_x', type=int, default=32)
    argument_parser.add_argument('--number_of_classes', type=int, default=16)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_x = arguments.number_of_x
    number_of_classes = arguments.number_of_classes
    path = os.path.join('./runs/', arguments.path)
    main(number_of_x, number_of_classes, path)
