from argparse import ArgumentParser
import os

import numpy as np
import torch
from torch_geometric.nn import VGAE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from encoder import Encoder
from negative_sampling import negative_sampling


def main(number_of_x, number_of_classes, epochs, path):
    dataset = Dataset(number_of_x, path)
    train_dataset = dataset[:dataset.number_of_train]
    model = VGAE(Encoder(number_of_x, number_of_classes))
    model.train()
    model = train(train_dataset, model, epochs, path)
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))


def train(dataset, model, epochs, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        for epoch in tqdm(range(epochs)):
            losses = aucs = aps = []
            with tqdm(dataset) as pbar:
                for data in pbar:
                    edge_index, x = data['edge_index'], data['x']
                    mu, logstd = model.encoder(x, edge_index)
                    z = model.encode(x, edge_index)
                    negative_edge_index = negative_sampling(edge_index, z.size(0))
                    loss = model.recon_loss(z, edge_index, negative_edge_index) + model.kl_loss(mu, logstd)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    model.eval()
                    auc, ap = model.test(z, edge_index, negative_edge_index)
                    model.train()
                    losses.append(loss.item())
                    aucs.append(auc)
                    aps.append(ap)
                    pbar.set_postfix_str('epoch: {}, loss: {:.3f}, auc: {:.3f}, ap: {:.3f}'.format(epoch, np.mean(np.array(losses)), np.mean(np.array(aucs)), np.mean(np.array(aps))))
            summary_writer.add_scalar('Loss/Train', np.mean(np.array(losses)), epoch)
            summary_writer.add_scalar('AUC/Train', np.mean(np.array(aucs)), epoch)
            summary_writer.add_scalar('AP/Train', np.mean(np.array(aps)), epoch)
    return model


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_features', type=int, default=32)
    argument_parser.add_argument('--number_of_classes', type=int, default=16)
    argument_parser.add_argument('--epochs', type=int, default=64)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_features = arguments.number_of_features
    number_of_classes = arguments.number_of_classes
    epochs = arguments.epochs
    path = os.path.join('./runs/', arguments.path)
    main(number_of_features, number_of_classes, epochs, path)
