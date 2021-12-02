from argparse import ArgumentParser
import os
from shutil import rmtree


def main(mode):
    if mode == 'make_dataset':
        make_dataset()
    elif mode == 'train':
        train()
    elif mode == 'test':
        test()
    elif mode == 'clean_make_dataset':
        clean_make_dataset()
    elif mode == 'clean_train':
        clean_train()
    elif mode == 'clean_test':
        clean_test()


def make_dataset():
    os.makedirs('./dataset', exist_ok=True)


def train():
    os.makedirs('./models', exist_ok=True)


def test():
    os.makedirs('./figures/min_covers', exist_ok=True)
    os.makedirs('./figures/approx_min_covers', exist_ok=True)
    os.makedirs('./figures/tests', exist_ok=True)


def clean_make_dataset():
    try:
        rmtree('./dataset', ignore_errors=True)
    except:
        pass


def clean_train():
    try:
        rmtree('./models', ignore_errors=True)
        rmtree('./runs', ignore_errors=True)
    except:
        pass


def clean_test():
    try:
        rmtree('./figures/min_covers', ignore_errors=True)
        rmtree('./figures/approx_min_covers', ignore_errors=True)
        rmtree('./figures/tests', ignore_errors=True)
    except:
        pass


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--mode')
    arguments = argument_parser.parse_args()
    mode = arguments.mode
    main(mode)
