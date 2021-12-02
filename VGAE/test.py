from argparse import ArgumentParser
import os
from multiprocessing import Pool, freeze_support

from torch_geometric.nn import VGAE
import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm.contrib.concurrent import process_map

from dataset import Dataset, Graphs
from encoder import Encoder


def main(number_of_features, number_of_classes, path):
    dataset = Dataset(number_of_features, path)
    test_dataset = dataset[dataset.number_of_train:]
    model = VGAE(Encoder(number_of_features, number_of_classes))
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
    model.eval()
    test(test_dataset, model)
    graphs = Graphs(path)
    test_graphs = graphs[dataset.number_of_train:]
    graphs = []
    for graph in test_graphs:
        graph = dgl.to_networkx(graph)
        graph = nx.Graph(graph)
        graphs.append(graph)
    process_map(savefigure_A_wrapper, [(graph, os.path.join(path, 'figures/A/{}.jpg'.format(dataset.number_of_train + index))) for index, graph in enumerate(graphs)], max_workers=os.cpu_count()+1)


def test(dataset, model):
    aucs = np.array([])
    aps = np.array([])
    for data in dataset:
        x, edge_index = data['x'], data['edge_index']
        z = model.encode(x, edge_index)
        negative_edge_index = negative_sampling(edge_index, z.size(0))
        auc, ap = model.test(z, edge_index, negative_edge_index)
        aucs = np.append(aucs, auc)
        aps = np.append(aps, ap)
    print('Test AUC: {:.3f}, AP: {:.3f}'.format(np.average(aucs), np.average(aps)))


def negative_sampling(edge_index, number_of_nodes):
    adjacency = [[False for _ in range(number_of_nodes)] for _ in range(number_of_nodes)]
    for i in range(number_of_nodes):
        adjacency[i][i] = True
    for i in range(min(len(edge_index[0]), len(edge_index[1]))):
        adjacency[edge_index[0][i]][edge_index[1][i]] = True
    sources = np.array([], dtype=np.int64)
    destinations = np.array([], dtype=np.int64)
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if not adjacency[i][j]:
                sources = np.append(sources, i)
                destinations = np.append(destinations, j)
    sources = torch.from_numpy(sources)
    destinations = torch.from_numpy(destinations)
    negative_edge_index = torch.stack((sources, destinations))
    return negative_edge_index


def savefigure_A_wrapper(arguments):
    graph, path = arguments
    with Pool(1) as pool:
        pool.map(savefigure_A, [[graph, path]])


def savefigure_A(arguments):
    graph, path = arguments
    position = nx.circular_layout(graph)
    nodes_color = ['#333333'] * graph.number_of_nodes()
    nx.draw_networkx(graph, position, node_color=nodes_color, font_color='#ffffff')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_features', type=int, default=32)
    argument_parser.add_argument('--number_of_classes', type=int, default=16)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_features = arguments.number_of_features
    number_of_classes = arguments.number_of_classes
    path = os.path.join('./runs/', arguments.path)
    main(number_of_features, number_of_classes, path)
