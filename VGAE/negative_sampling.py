import numpy as np
import torch


def negative_sampling(edge_index, number_of_nodes):
    adjacency = [[False for _ in range(number_of_nodes)] for _ in range(number_of_nodes)]
    for i in range(number_of_nodes):
        adjacency[i][i] = True
    for i in range(min(len(edge_index[0]), len(edge_index[1]))):
        adjacency[edge_index[0][i]][edge_index[1][i]] = True
    sources = destinations = []
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if not adjacency[i][j]:
                sources.append(i)
                destinations.append(j)
    negative_edge_index = torch.stack((torch.from_numpy(np.array(sources, dtype=np.int64)), torch.from_numpy(np.array(destinations, dtype=np.int64))))
    return negative_edge_index
