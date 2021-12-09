from argparse import ArgumentParser
from multiprocessing import Pool, freeze_support
import os

import networkx as nx
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

from dataset import Dataset, Graphs


def main(number_of_x, path):
    dataset = Dataset(number_of_x, path)
    graphs = Graphs(path)
    graphs_test = graphs[dataset.number_of_train:]
    process_map(save_figure_wrapper, [(graph, os.path.join(path, 'figures/A/{}.jpg'.format(dataset.number_of_train + index))) for index, graph in enumerate(graphs_test)], max_workers=os.cpu_count() + 1)


def save_figure_wrapper(arguments):
    graph, path = arguments
    with Pool(1) as pool:
        pool.map(save_figure, [[graph, path]])


def save_figure(arguments):
    graph, path = arguments
    nodes_color = ['#333'] * graph.number_of_nodes()
    nx.draw_networkx(graph, nx.circular_layout(graph), node_color=nodes_color, font_color='#ffffff')
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
