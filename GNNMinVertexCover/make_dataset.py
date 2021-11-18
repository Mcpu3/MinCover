import os
from multiprocessing import Pool, freeze_support
import random
import sys
import networkx as nx
import pandas
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


def main():
    n_gs = 1000
    gs = []
    for g_id in tqdm(range(n_gs)):
        n = 16
        p = min(max(random.random(), 0.1), 0.25)
        g = nx.fast_gnp_random_graph(n, p)
        gs.append(g)
    min_covers = process_map(min_vertex_cover_wrapper, [
                             (g,) for g in gs], max_workers=os.cpu_count()+1)
    for g_id, g in enumerate(tqdm(gs)):
        d_nodes = {'g_id': [], 'nodes': [], 'labels': []}
        d_nodes['g_id'] = [g_id for _ in range(g.number_of_nodes())]
        d_nodes['nodes'] = [node for node in range(g.number_of_nodes())]
        d_nodes['labels'] = [0 for _ in range(g.number_of_nodes())]
        for node in min_covers[g_id]:
            d_nodes['labels'][node] = 1
        df_nodes = pandas.DataFrame(d_nodes)
        if g_id == 0:
            df_nodes.to_csv('./dataset/nodes.csv', index=False)
        else:
            df_nodes.to_csv('./dataset/nodes.csv',
                            header=False, index=False, mode='a')
        d_edges = {'g_id': [], 'src': [], 'dst': []}
        d_edges['g_id'] = [g_id for _ in range(g.number_of_edges() * 2)]
        for edge in g.edges():
            d_edges['src'].append(edge[0])
            d_edges['dst'].append(edge[1])
            d_edges['src'].append(edge[1])
            d_edges['dst'].append(edge[0])
        df_edges = pandas.DataFrame(d_edges)
        if g_id == 0:
            df_edges.to_csv('./dataset/edges.csv', index=False)
        else:
            df_edges.to_csv('./dataset/edges.csv',
                            header=False, index=False, mode='a')


def min_vertex_cover_wrapper(args):
    g = args[0]
    with Pool(1) as p:
        min_cover = p.map(min_vertex_cover, [[g]])
    return min_cover[0]


def min_vertex_cover(args):
    g = args[0]
    min_cover = set()
    min_weight = sys.maxsize
    for i in range(2 ** g.number_of_nodes()):
        nodes = set()
        edges = set()
        for j in range(g.number_of_nodes()):
            if (i >> j) & 1:
                nodes.add(j)
                for k in g.adj[j]:
                    if j < k:
                        edges.add((j, k))
                    else:
                        edges.add((k, j))
        if edges == set(g.edges()):
            if len(nodes) < min_weight:
                min_cover = nodes
                min_weight = len(nodes)
    return min_cover


if __name__ == '__main__':
    freeze_support()
    main()
