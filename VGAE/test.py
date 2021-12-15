from argparse import ArgumentParser
import os

import torch
from torch_geometric.nn import VGAE
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from encoder import Encoder


def main(number_of_x, number_of_classes, path):
    dataset = Dataset(number_of_x, path)
    model = VGAE(Encoder(number_of_x, number_of_classes))
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
    model.eval()
    test(dataset, model, path, dataset.number_of_train)


def test(dataset, model, path, number_of_train):
    with SummaryWriter(os.path.join(path, 'runs/')) as summary_writer:
        for index, data in enumerate(tqdm(dataset)):
            edge_index, x = data['edge_index'], data['x']
            z = model.encode(x, edge_index)
            negative_edge_index = negative_sampling(edge_index, z.size(0))
            auc, ap = model.test(z, edge_index, negative_edge_index)
            summary_writer.add_scalar('AUC/Test', auc, number_of_train + index)
            summary_writer.add_scalar('AP/Test', ap, number_of_train + index)


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--number_of_x', type=int, default=32)
    argument_parser.add_argument('--number_of_classes', type=int, default=16)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    number_of_x = arguments.number_of_x
    number_of_classes = arguments.number_of_classes
    path = os.path.join('./runs/', arguments.path)
    main(number_of_x, number_of_classes, path)
