from argparse import ArgumentParser
import os


def main(clean, path):
    if not clean:
        os.mkdir(path)
        make_dataset(path)
        train(path)
        test(path)
    else:
        clean_make_dataset(path)
        clean_train(path)
        clean_test(path)
        os.rmdir(path)


def make_dataset(path):
    os.mkdir(os.path.join(path, 'dataset/'))


def train(path):
    os.mkdir(os.path.join(path, 'runs/'))


def test(path):
    os.mkdir(os.path.join(path, 'figures/'))
    os.mkdir(os.path.join(path, 'figures/min_covers/'))
    os.mkdir(os.path.join(path, 'figures/approx_min_covers/'))
    os.mkdir(os.path.join(path, 'figures/tests/'))


def clean_make_dataset(path):
    for dir in os.listdir(os.path.join(path, 'dataset/')):
        os.remove(os.path.join(os.path.join(path, 'dataset/'), dir))
    os.rmdir(os.path.join(path, 'dataset/'))


def clean_train(path):
    for dir in os.listdir(os.path.join(path, 'runs/')):
        os.remove(os.path.join(os.path.join(path, 'runs/'), dir))
    os.rmdir(os.path.join(path, 'runs/'))


def clean_test(path):
    for dir in os.listdir(os.path.join(path, 'figures/min_covers/')):
        os.remove(os.path.join(os.path.join(path, 'figures/min_covers/'), dir))
    for dir in os.listdir(os.path.join(path, 'figures/approx_min_covers/')):
        os.remove(os.path.join(os.path.join(path, 'figures/approx_min_covers/'), dir))
    for dir in os.listdir(os.path.join(path, 'figures/tests/')):
        os.remove(os.path.join(os.path.join(path, 'figures/tests/'), dir))
    os.rmdir(os.path.join(path, 'figures/min_covers/'))
    os.rmdir(os.path.join(path, 'figures/approx_min_covers/'))
    os.rmdir(os.path.join(path, 'figures/tests/'))
    os.rmdir(os.path.join(path, 'figures/'))


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--clean', action='store_true')
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    clean = arguments.clean
    path = os.path.join('./runs/', arguments.path)
    main(clean, path)
