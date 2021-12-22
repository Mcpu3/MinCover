import copy
import sys

import networkx as nx


def min_vertex_cover(arguments):
    graph = arguments[0]
    min_cover = set()
    min_weight = sys.maxsize
    nodes_of_graph = list(graph.nodes())
    for i in range(2 ** graph.number_of_nodes()):
        nodes, edges = set(), set()
        for j in range(graph.number_of_nodes()):
            if (i >> j) & 1:
                nodes.add(nodes_of_graph[j])
                for k in graph.adj[nodes_of_graph[j]]:
                    if nodes_of_graph[j] < k:
                        edges.add((nodes_of_graph[j], k))
                    else:
                        edges.add((k, nodes_of_graph[j]))
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
    min_cover_copy = copy.deepcopy(min_cover)
    min_cover = set()
    for node in range(len(min_cover_copy)):
        if min_cover_copy[node]:
            min_cover.add(node)
    return min_cover


# 0個の辺を持つ頂点をmin_coverから削除
def min_vertex_cover_with_supervised_learning_1(arguments):
    data, model = arguments
    x = data.ndata['x']
    min_cover = model(data, x).argmax(1)
    min_cover_copy = copy.deepcopy(min_cover)
    min_cover = set()
    for node in range(len(min_cover_copy)):
        if min_cover_copy[node]:
            min_cover.add(node)
    min_cover_copy = copy.deepcopy(min_cover)
    for node in min_cover_copy:
        if len(data.successors(node)) == 1:
            min_cover.remove(node)
    return min_cover


# 0個の辺を持つ頂点をmin_coverから削除, 接続された頂点がmin_coverに無い辺に接続された頂点で構成された部分グラフにおいて最適解を求解
def min_vertex_cover_with_supervised_learning_2(arguments):
    data, model = arguments
    x = data.ndata['x']
    min_cover = model(data, x).argmax(1)
    min_cover_copy = copy.deepcopy(min_cover)
    min_cover = set()
    for node in range(len(min_cover_copy)):
        if min_cover_copy[node]:
            min_cover.add(node)
    min_cover_copy = copy.deepcopy(min_cover)
    for node in min_cover_copy:
        if len(data.successors(node)) == 1:
            min_cover.remove(node)
    sub_graph = nx.Graph()
    edges_of_sub_graph = []
    for edge_source, edge_destination in zip(data.edges()[0], data.edges()[1]):
        if (edge_source.item() not in min_cover) and (edge_destination.item() not in min_cover):
            edges_of_sub_graph.append((edge_source.item(), edge_destination.item()))
    sub_graph.add_edges_from(edges_of_sub_graph)
    sub_graph.remove_edges_from(nx.selfloop_edges(sub_graph))
    min_cover_of_sub_graph = min_vertex_cover((sub_graph,))
    min_cover = min_cover.union(min_cover_of_sub_graph)
    return min_cover
