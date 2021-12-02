import os

from torch_geometric.data import Data
import dgl
import pandas as pd
import torch


class Dataset():
    def __init__(self, number_of_features, path):
        self.dataset = []
        nodes = pd.read_csv(os.path.join(path, 'dataset/nodes.csv'))
        edges = pd.read_csv(os.path.join(path, 'dataset/edges.csv'))
        nodes_group = nodes.groupby('graph_id')
        edges_group = edges.groupby('graph_id')
        for graph_id in nodes_group.groups:
            nodes_of_id = nodes_group.get_group(graph_id)
            number_of_nodes = nodes_of_id.to_numpy().shape[0]
            sourses = torch.empty(0, dtype=torch.int64)
            destinations = torch.empty(0, dtype=torch.int64)
            if graph_id in edges_group.groups.keys():
                edges_of_id = edges_group.get_group(graph_id)
                sourses = torch.from_numpy(edges_of_id['sources'].to_numpy())
                destinations = torch.from_numpy(edges_of_id['destinations'].to_numpy())
            x = torch.rand([number_of_nodes, number_of_features], dtype=torch.float32)
            edge_index = torch.stack((sourses, destinations))
            data = Data(x, edge_index)
            self.dataset.append(data)
        self.number_of_train = int(len(self.dataset) * 0.5)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class Graphs():
    def __init__(self, path):
        self.graphs = []
        nodes = pd.read_csv(os.path.join(path, 'dataset/nodes.csv'))
        edges = pd.read_csv(os.path.join(path, 'dataset/edges.csv'))
        nodes_group = nodes.groupby('graph_id')
        edges_group = edges.groupby('graph_id')
        for graph_id in nodes_group.groups:
            nodes_of_id = nodes_group.get_group(graph_id)
            number_of_nodes = nodes_of_id.to_numpy().shape[0]
            sourses = torch.empty(0, dtype=torch.int64)
            destinations = torch.empty(0, dtype=torch.int64)
            if graph_id in edges_group.groups.keys():
                edges_of_id = edges_group.get_group(graph_id)
                sourses = torch.from_numpy(edges_of_id['sources'].to_numpy())
                destinations = torch.from_numpy(edges_of_id['destinations'].to_numpy())
            graph = dgl.graph((sourses, destinations), num_nodes=number_of_nodes)
            self.graphs.append(graph)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)