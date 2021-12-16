from argparse import ArgumentParser
from multiprocessing import Pool, freeze_support
import os
import random

import networkx as nx
import pandas
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from min_vertex_cover import min_vertex_cover


def main(number_of_graphs, number_of_nodes, p_min, p_max, without_label, path):
    graphs = []
    for graph_id in tqdm(range(number_of_graphs)):
        p = min(max(random.random(), p_min), p_max)
        graph = nx.fast_gnp_random_graph(number_of_nodes, p)
        graphs.append(graph)
    min_covers = [set() for _ in range(len(graphs))]
    if not without_label:
        min_covers = process_map(min_cover_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count() + 1)
    for graph_id, graph in enumerate(tqdm(graphs)):
        nodes = {'graph_id': [], 'nodes': [], 'label': []}
        nodes['graph_id'] = [graph_id for _ in range(graph.number_of_nodes())]
        nodes['nodes'] = [node for node in range(graph.number_of_nodes())]
        nodes['label'] = [0 for _ in range(graph.number_of_nodes())]
        for node in min_covers[graph_id]:
            nodes['label'][node] = 1
        nodes = pandas.DataFrame(nodes)
        if graph_id == 0:
            nodes.to_csv(os.path.join(path, 'dataset/nodes.csv'), index=False)
        else:
            nodes.to_csv(os.path.join(path, 'dataset/nodes.csv'), header=False, index=False, mode='a')
        edges = {'graph_id': [], 'sourses': [], 'destinations': []}
        edges['graph_id'] = [graph_id for _ in range(graph.number_of_edges() * 2)]
        for edge in graph.edges():
            edges['sourses'].append(edge[0])
            edges['destinations'].append(edge[1])
            edges['sourses'].append(edge[1])
            edges['destinations'].append(edge[0])
        edges = pandas.DataFrame(edges)
        if graph_id == 0:
            edges.to_csv(os.path.join(path, 'dataset/edges.csv'), index=False)
        else:
            edges.to_csv(os.path.join(path, 'dataset/edges.csv'), header=False, index=False, mode='a')


def min_cover_wrapper(arguments):
    graphs = arguments[0]
    with Pool(1) as pool:
        min_cover = pool.map(min_vertex_cover, [[graphs]])[0]
    return min_cover


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_graphs', type=int, default=1000)
    argument_parser.add_argument('--number_of_nodes', type=int, default=16)
    argument_parser.add_argument('--p_min', type=float, default=0.1)
    argument_parser.add_argument('--p_max', type=float, default=0.9)
    argument_parser.add_argument('--without_label', action='store_true')
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_graphs = arguments.number_of_graphs
    number_of_nodes = arguments.number_of_nodes
    p_min = arguments.p_min
    p_max = arguments.p_max
    without_label = arguments.without_label
    path = os.path.join('./runs/', arguments.path)
    main(number_of_graphs, number_of_nodes, p_min, p_max, without_label, path)
