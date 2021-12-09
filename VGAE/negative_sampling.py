import numpy as np
import torch


def negative_sampling(edge_index, number_of_nodes):
    adjacency = [[False for _ in range(number_of_nodes)] for _ in range(number_of_nodes)]
    for i in range(number_of_nodes):
        adjacency[i][i] = True
    for i in range(min(len(edge_index[0]), len(edge_index[1]))):
        adjacency[edge_index[0][i]][edge_index[1][i]] = True
    sources = np.array([], dtype=np.int64)
    destinations = np.array([], dtype=np.int64)
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if not adjacency[i][j]:
                sources = np.append(sources, i)
                destinations = np.append(destinations, j)
    negative_edge_index = torch.stack((torch.from_numpy(sources), torch.from_numpy(destinations)))
    return negative_edge_index
