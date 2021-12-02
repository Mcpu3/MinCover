from argparse import ArgumentParser

import os
from multiprocessing import Pool, freeze_support
import random
import sys
import networkx as nx
import pandas
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


def main(number_of_graphs, number_of_nodes, p_min, p_max, path):
    graphs = []
    for graph_id in tqdm(range(number_of_graphs)):
        p = min(max(random.random(), p_min), p_max)
        graph = nx.fast_gnp_random_graph(number_of_nodes, p)
        graphs.append(graph)
    min_covers = process_map(min_vertex_cover_wrapper, [(graph,) for graph in graphs], max_workers=os.cpu_count()+1)
    for graph_id, graph in enumerate(tqdm(graphs)):
        d_nodes = {'graph_id': [], 'nodes': [], 'labels': []}
        d_nodes['graph_id'] = [graph_id for _ in range(graph.number_of_nodes())]
        d_nodes['nodes'] = [node for node in range(graph.number_of_nodes())]
        d_nodes['labels'] = [0 for _ in range(graph.number_of_nodes())]
        for node in min_covers[graph_id]:
            d_nodes['labels'][node] = 1
        df_nodes = pandas.DataFrame(d_nodes)
        if graph_id == 0:
            df_nodes.to_csv(os.path.join(path, 'dataset/nodes.csv'), index=False)
        else:
            df_nodes.to_csv(os.path.join(path, 'dataset/nodes.csv'), header=False, index=False, mode='a')
        d_edges = {'graph_id': [], 'sourses': [], 'destinations': []}
        d_edges['graph_id'] = [graph_id for _ in range(graph.number_of_edges() * 2)]
        for edge in graph.edges():
            d_edges['sourses'].append(edge[0])
            d_edges['destinations'].append(edge[1])
            d_edges['sourses'].append(edge[1])
            d_edges['destinations'].append(edge[0])
        df_edges = pandas.DataFrame(d_edges)
        if graph_id == 0:
            df_edges.to_csv(os.path.join(path, 'dataset/edges.csv'), index=False)
        else:
            df_edges.to_csv(os.path.join(path, 'dataset/edges.csv'), header=False, index=False, mode='a')


def min_vertex_cover_wrapper(arguments):
    graphs = arguments[0]
    with Pool(1) as pool:
        min_cover = pool.map(min_vertex_cover, [[graphs]])
    return min_cover[0]


def min_vertex_cover(arguments):
    graphs = arguments[0]
    min_cover = set()
    min_weight = sys.maxsize
    for i in range(2 ** graphs.number_of_nodes()):
        nodes = set()
        edges = set()
        for j in range(graphs.number_of_nodes()):
            if (i >> j) & 1:
                nodes.add(j)
                for k in graphs.adj[j]:
                    if j < k:
                        edges.add((j, k))
                    else:
                        edges.add((k, j))
        if edges == set(graphs.edges()):
            if len(nodes) < min_weight:
                min_cover = nodes
                min_weight = len(nodes)
    return min_cover


if __name__ == '__main__':
    freeze_support()
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_graphs', type=int, default=1000)
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
