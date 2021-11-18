import os
from multiprocessing import Pool, freeze_support
import random
import matplotlib.pyplot as plt
import networkx as nx
import pandas
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


def main():
    n_gs = 2
    gs = []
    for g_id in tqdm(range(n_gs)):
        n = 256
        p = min(max(random.random(), 0.1), 0.9)
        g = nx.fast_gnp_random_graph(n, p)
        d_nodes = {'g_id': [], 'nodes': []}
        d_nodes['g_id'] = [g_id] * g.number_of_nodes()
        d_nodes['nodes'] = [node for node in range(g.number_of_nodes())]
        df_nodes = pandas.DataFrame(d_nodes)
        if g_id == 0:
            df_nodes.to_csv('./dataset/nodes.csv', index=False)
        else:
            df_nodes.to_csv('./dataset/nodes.csv',
                            header=False, index=False, mode='a')
        d_edges = {'g_id': [], 'src': [], 'dst': []}
        d_edges['g_id'] = [g_id] * (g.number_of_edges() * 2)
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
        gs.append(g)
    process_map(savefig_A_wrapper, [(g, './fig/A/{}.jpg'.format(g_id))
                for g, g_id in zip(gs, range(n_gs))], max_workers=os.cpu_count()+1)


def savefig_A_wrapper(args):
    g, path = args
    with Pool(1) as p:
        p.map(savefig_A, [[g, path]])


def savefig_A(args):
    g, path = args
    pos = nx.circular_layout(g)
    node_color = ['#333333'] * g.number_of_nodes()
    nx.draw_networkx(g, pos, node_color=node_color, font_color='#ffffff')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


if __name__ == '__main__':
    freeze_support()
    main()
