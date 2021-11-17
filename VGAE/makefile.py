from argparse import ArgumentParser
import os
from shutil import rmtree


def main(mode):
    if mode == 'make_dataset':
        make_dataset()
    elif mode == 'train':
        train()
    elif mode == 'clean_make_dataset':
        clean_make_dataset()
    elif mode == 'clean_train':
        clean_train()


def make_dataset():
    os.makedirs('./dataset', exist_ok=True)
    os.makedirs('./fig/A', exist_ok=True)


def train():
    os.makedirs('./fig/A_tilda', exist_ok=True)
    os.makedirs('./models', exist_ok=True)


def clean_make_dataset():
    try:
        rmtree('./dataset', ignore_errors=True)
        rmtree('./fig/A', ignore_errors=True)
    except:
        pass


def clean_train():
    try:
        rmtree('./fig/A_tilda', ignore_errors=True)
        rmtree('./models', ignore_errors=True)
        rmtree('./runs', ignore_errors=True)
    except:
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode')
    args = parser.parse_args()
    main(args.mode)
