import os

import dgl
from dgl.data import DGLDataset
import pandas as pd
import torch


class Dataset(DGLDataset):
    def __init__(self, number_of_features, path):
        self.number_of_features = number_of_features
        self.path = path
        super().__init__(name='min_vertex_cover')

    def process(self):
        self.number_of_classes = 2
        self.graphs = []
        nodes = pd.read_csv(os.path.join(self.path, 'dataset/nodes.csv'))
        edges = pd.read_csv(os.path.join(self.path, 'dataset/edges.csv'))
        nodes_group = nodes.groupby('graph_id')
        edges_group = edges.groupby('graph_id')
        for graph_id in nodes_group.groups:
            nodes_of_id = nodes_group.get_group(graph_id)
            number_of_nodes = nodes_of_id.to_numpy().shape[0]
            labels = torch.from_numpy(nodes_of_id['labels'].to_numpy())
            sourses = torch.empty(0, dtype=torch.int64)
            destinations = torch.empty(0, dtype=torch.int64)
            if graph_id in edges_group.groups.keys():
                edges_of_id = edges_group.get_group(graph_id)
                sourses = torch.from_numpy(edges_of_id['sourses'].to_numpy())
                destinations = torch.from_numpy(
                    edges_of_id['destinations'].to_numpy())
            graph = dgl.graph((sourses, destinations),
                              num_nodes=number_of_nodes)
            graph = dgl.add_self_loop(graph)
            features = torch.rand(
                [number_of_nodes, self.number_of_features], dtype=torch.float32)
            graph.ndata['features'] = features
            graph.ndata['labels'] = labels
            self.graphs.append(graph)
        self.number_of_train = int(len(self.graphs) * 0.5)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)