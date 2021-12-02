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
        d_nodes = {'graph_id': [], 'nodes': []}
        d_nodes['graph_id'] = [graph_id] * graph.number_of_nodes()
        d_nodes['nodes'] = [node for node in range(graph.number_of_nodes())]
        df_nodes = pandas.DataFrame(d_nodes)
        if graph_id == 0:
            df_nodes.to_csv(os.path.join(path, 'dataset/nodes.csv'), index=False)
        else:
            df_nodes.to_csv(os.path.join(path, 'dataset/nodes.csv'), header=False, index=False, mode='a')
        d_edges = {'graph_id': [], 'sources': [], 'destinations': []}
        d_edges['graph_id'] = [graph_id] * (graph.number_of_edges() * 2)
        for edge in graph.edges():
            d_edges['sources'].append(edge[0])
            d_edges['destinations'].append(edge[1])
            d_edges['sources'].append(edge[1])
            d_edges['destinations'].append(edge[0])
        df_edges = pandas.DataFrame(d_edges)
        if graph_id == 0:
            df_edges.to_csv(os.path.join(path, 'dataset/edges.csv'), index=False)
        else:
            df_edges.to_csv(os.path.join(path, 'dataset/edges.csv'), header=False, index=False, mode='a')


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_graphs', type=int, default=2)
    argument_parser.add_argument('--number_of_nodes', type=int, default=16)
    argument_parser.add_argument('--p_min', type=float, default=0.1)
    argument_parser.add_argument('--p_max', type=float, default=0.9)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_graphs = arguments.number_of_graphs
    number_of_nodes = arguments.number_of_nodes
    p_min = arguments.p_min
    p_max = arguments.p_max
    path = os.path.join('./runs/', arguments.path)
    main(number_of_graphs, number_of_nodes, p_min, p_max, path)
