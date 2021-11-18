import os
from multiprocessing import Pool, freeze_support
import random
import sys
import matplotlib.pyplot as plt
import networkx as nx
import pandas
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


def main():
    n_gs = 1000
    gs = []
    for g_id in tqdm(range(n_gs)):
        n = 16
        p = min(max(random.random(), 0.1), 0.9)
        g = nx.fast_gnp_random_graph(n, p)
        gs.append(g)
    min_covers = process_map(min_vertex_cover_wrapper, [
                             (g,) for g in gs], max_workers=os.cpu_count()+1)
    approx_min_covers = process_map(approx_min_vertex_cover_wrapper, [
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
    process_map(savefig_min_cover_wrapper, [(g, min_cover, './fig/min_covers/{}.jpg'.format(g_id))
                for g, min_cover, g_id in zip(gs, min_covers, range(n_gs))], max_workers=os.cpu_count()+1)
    process_map(savefig_min_cover_wrapper, [(g, approx_min_cover, './fig/approx_min_covers/{}.jpg'.format(g_id))
                for g, approx_min_cover, g_id in zip(gs, approx_min_covers, range(n_gs))], max_workers=os.cpu_count()+1)


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


def approx_min_vertex_cover_wrapper(args):
    g = args[0]
    with Pool(1) as p:
        min_cover = p.map(approx_min_vertex_cover, [[g]])
    return min_cover[0]


def approx_min_vertex_cover(args):
    g = args[0]
    min_cover = nx.algorithms.approximation.min_weighted_vertex_cover(g)
    return min_cover


def savefig_min_cover_wrapper(args):
    g, min_cover, path = args
    with Pool(1) as p:
        p.map(savefig_min_cover, [[g, min_cover, path]])


def savefig_min_cover(args):
    g, min_cover, path = args
    pos = nx.circular_layout(g)
    node_color = ['#333333'] * g.number_of_nodes()
    for node in min_cover:
        node_color[node] = '#009b9f'
    nx.draw_networkx(g, pos, node_color=node_color, font_color='#ffffff')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


if __name__ == '__main__':
    freeze_support()
    main()
