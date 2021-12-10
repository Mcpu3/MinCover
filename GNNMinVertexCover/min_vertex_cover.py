import sys

import networkx as nx


def min_vertex_cover(arguments):
    graph = arguments[0]
    min_cover = set()
    min_weight = sys.maxsize
    for i in range(2 ** graph.number_of_nodes()):
        nodes, edges = set(), set()
        for j in range(graph.number_of_nodes()):
            if (i >> j) & 1:
                nodes.add(j)
                for k in graph.adj[j]:
                    if j < k:
                        edges.add((j, k))
                    else:
                        edges.add((k, j))
        if edges == set(graph.edges()):
            if len(nodes) < min_weight:
                min_cover = nodes
                min_weight = len(nodes)
    return min_cover


def min_vertex_cover_approx(arguments):
    graph = arguments[0]
    min_cover = nx.algorithms.approximation.min_weighted_vertex_cover(graph)
    return min_cover


def min_vertex_cover_with_supervised_learning(arguments):
    data, model = arguments
    x = data.ndata['x']
    min_cover = model(data, x).argmax(1)
    min_cover_copy = min_cover
    min_cover = set()
    for node in range(len(min_cover_copy)):
        if min_cover_copy[node]:
            min_cover.add(node)
    return min_cover
