import os

import dgl
from dgl.data import DGLDataset
import networkx as nx
import pandas as pd
import torch


class Dataset(DGLDataset):
    def __init__(self, number_of_x, path):
        self.number_of_x = number_of_x
        self.path = path
        super().__init__(name='min_cover')

    def process(self):
        self.number_of_classes = 2
        self.dataset = []
        nodes = pd.read_csv(os.path.join(self.path, 'dataset/nodes.csv')).groupby('graph_id')
        edges = pd.read_csv(os.path.join(self.path, 'dataset/edges.csv')).groupby('graph_id')
        for graph_id in nodes.groups:
            nodes_of_graph_id = nodes.get_group(graph_id)
            number_of_nodes = nodes_of_graph_id.to_numpy().shape[0]
            label = torch.from_numpy(nodes_of_graph_id['label'].to_numpy())
            sources, destinations = torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
            if graph_id in edges.groups.keys():
                edges_of_graph_id = edges.get_group(graph_id)
                sources = torch.from_numpy(edges_of_graph_id['sourses'].to_numpy())
                destinations = torch.from_numpy(edges_of_graph_id['destinations'].to_numpy())
            data = dgl.add_self_loop(dgl.graph((sources, destinations), num_nodes=number_of_nodes))
            x = torch.rand([number_of_nodes, self.number_of_x], dtype=torch.float32)
            data.ndata['label'], data.ndata['x'] = label, x
            self.dataset.append(data)
        self.number_of_train = int(len(self.dataset) * 0.5)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class Graphs():
    def __init__(self, path):
        self.graphs = []
        nodes = pd.read_csv(os.path.join(path, 'dataset/nodes.csv')).groupby('graph_id')
        edges = pd.read_csv(os.path.join(path, 'dataset/edges.csv')).groupby('graph_id')
        for graph_id in nodes.groups:
            nodes_of_graph_id = nodes.get_group(graph_id)
            number_of_nodes = nodes_of_graph_id.to_numpy().shape[0]
            sources, destinations = torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
            if graph_id in edges.groups.keys():
                edges_of_graph_id = edges.get_group(graph_id)
                sources = torch.from_numpy(edges_of_graph_id['sourses'].to_numpy())
                destinations = torch.from_numpy(edges_of_graph_id['destinations'].to_numpy())
            graph = nx.Graph(dgl.to_networkx(dgl.graph((sources, destinations), num_nodes=number_of_nodes)))
            self.graphs.append(graph)

    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.graphs)
