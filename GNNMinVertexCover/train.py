from argparse import ArgumentParser
import os

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from gcn import GCN


def main(number_of_features, epochs, path):
    dataset = Dataset(number_of_features, path)
    train_dataset = dataset[:dataset.number_of_train]
    model = GCN(dataset.number_of_features, dataset.number_of_classes)
    model.train()
    model = train(train_dataset, model, epochs, path)
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))


def train(dataset, model, epochs, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        for epoch in tqdm(range(epochs)):
            losses = np.array([])
            accs = np.array([])
            aucs = np.array([])
            aps = np.array([])
            with tqdm(dataset) as pbar:
                for graph in pbar:
                    features = graph.ndata['features']
                    labels = graph.ndata['labels']
                    logits = model(graph, features)
                    loss = F.cross_entropy(logits, labels)
                    losses = np.append(losses, loss.item())
                    predicts = logits.argmax(1)
                    acc = sklearn.metrics.accuracy_score(labels, predicts)
                    accs = np.append(accs, acc)
                    auc = sklearn.metrics.roc_auc_score(labels, predicts)
                    aucs = np.append(aucs, auc)
                    ap = sklearn.metrics.average_precision_score(
                        labels, predicts)
                    aps = np.append(aps, ap)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix_str('epoch: {}, loss: {:.3f}, acc: {:.3f}, auc: {:.3f}, ap: {:.3f}'.format(epoch, np.average(losses), np.average(accs), np.average(aucs), np.average(aps)))
            summary_writer.add_scalar('Loss/Train', np.average(losses), epoch)
            summary_writer.add_scalar('Acc/Train', np.average(accs), epoch)
            summary_writer.add_scalar('AUC/Train', np.average(aucs), epoch)
            summary_writer.add_scalar('AP/Train', np.average(aps), epoch)
    return model


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_features', type=int, default=32)
    argument_parser.add_argument('--epochs', type=int, default=64)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_features = arguments.number_of_features
    epochs = arguments.epochs
    path = os.path.join('./runs/', arguments.path)
    main(number_of_features, epochs, path)
