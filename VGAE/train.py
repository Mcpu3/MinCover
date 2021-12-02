from argparse import ArgumentParser
import os

from torch_geometric.nn import VGAE
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from encoder import Encoder


def main(number_of_features, number_of_classes, epochs, path):
    dataset = Dataset(number_of_features, path)
    train_dataset = dataset[:dataset.number_of_train]
    model = VGAE(Encoder(number_of_features, number_of_classes))
    model.train()
    model = train(train_dataset, model, epochs, path)
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))


def train(dataset, model, epochs, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with SummaryWriter(os.path.join(path, 'runs/')) as writer:
        for epoch in tqdm(range(epochs)):
            aucs = np.array([])
            aps = np.array([])
            with tqdm(dataset) as pbar:
                for data in pbar:
                    x, edge_index = data['x'], data['edge_index']
                    mu, logstd = model.encoder(x, edge_index)
                    z = model.encode(x, edge_index)
                    neg_edge_index = negative_sampling(edge_index, z.size(0))
                    loss = model.recon_loss(z, edge_index, neg_edge_index) + \
                        model.kl_loss(mu, logstd)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    model.eval()
                    auc, ap = model.test(z, edge_index, neg_edge_index)
                    model.train()
                    aucs = np.append(aucs, auc)
                    aps = np.append(aps, ap)
                    pbar.set_postfix_str(
                        'epoch: {}, loss: {:.3f}, auc: {:.3f}, ap: {:.3f}'.format(epoch, loss, np.average(aucs), np.average(aps)))
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('AUC/train', np.average(aucs), epoch)
            writer.add_scalar('AP/train', np.average(aps), epoch)
    return model


def negative_sampling(edge_index, number_of_nodes):
    adjacency = [[False for _ in range(number_of_nodes)]
                 for _ in range(number_of_nodes)]
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
    sources = torch.from_numpy(sources)
    destinations = torch.from_numpy(destinations)
    negative_edge_index = torch.stack((sources, destinations))
    return negative_edge_index


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_features', type=int, default=32)
    argument_parser.add_argument('--number_of_classes', type=int, default=16)
    argument_parser.add_argument('--epochs', type=int, default=64)
    argument_parser.add_argument('--path', default='')
    arguments = argument_parser.parse_args()
    number_of_features = arguments.number_of_features
    number_of_classes = arguments.number_of_classes
    epochs = arguments.epochs
    path = os.path.join('./runs/', arguments.path)
    main(number_of_features, number_of_classes, epochs, path)
