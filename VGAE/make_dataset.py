from argparse import ArgumentParser
import os
import random

import networkx as nx
import pandas
from tqdm import tqdm


def main(number_of_graphs, number_of_nodes, p_min, p_max, path):
    for graph_id in tqdm(range(number_of_graphs)):
        p = min(max(random.random(), p_min), p_max)
        graph = nx.fast_gnp_random_graph(number_of_nodes, p)
        nodes = {'graph_id': [], 'nodes': []}
        nodes['graph_id'] = [graph_id for _ in range(graph.number_of_nodes())]
        nodes['nodes'] = [node for node in range(graph.number_of_nodes())]
        nodes = pandas.DataFrame(nodes)
        if graph_id == 0:
            nodes.to_csv(os.path.join(path, 'dataset/nodes.csv'), index=False)
        else:
            nodes.to_csv(os.path.join(path, 'dataset/nodes.csv'), header=False, index=False, mode='a')
        edges = {'graph_id': [], 'sources': [], 'destinations': []}
        edges['graph_id'] = [graph_id for _ in range(graph.number_of_edges() * 2)]
        for edge in graph.edges():
            edges['sources'].append(edge[0])
            edges['destinations'].append(edge[1])
            edges['sources'].append(edge[1])
            edges['destinations'].append(edge[0])
        edges = pandas.DataFrame(edges)
        if graph_id == 0:
            edges.to_csv(os.path.join(path, 'dataset/edges.csv'), index=False)
        else:
            edges.to_csv(os.path.join(path, 'dataset/edges.csv'), header=False, index=False, mode='a')


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_graphs', type=int, default=1)
    argument_parser.add_argument('--number_of_nodes', type=int, default=1024)
    argument_parser.add_argument('--p_min', type=float, default=0.1)
    argument_parser.add_argument('--p_max', type=float, default=0.1)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_graphs = arguments.number_of_graphs
    number_of_nodes = arguments.number_of_nodes
    p_min = arguments.p_min
    p_max = arguments.p_max
    path = os.path.join('./runs/', arguments.path)
    main(number_of_graphs, number_of_nodes, p_min, p_max, path)
