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
    os.mkdir(os.path.join(path, 'figures/with_supervised_learning/'))
    os.mkdir(os.path.join(path, 'figures/with_supervised_learning_1/'))
    os.mkdir(os.path.join(path, 'figures/with_supervised_learning_2/'))
    os.mkdir(os.path.join(path, 'figures/boxes/'))


def clean_make_dataset(path):
    for directory in os.listdir(os.path.join(path, 'dataset/')):
        os.remove(os.path.join(os.path.join(path, 'dataset/'), directory))
    os.rmdir(os.path.join(path, 'dataset/'))


def clean_train(path):
    os.remove(os.path.join(path, 'model.pth'))
    for directory in os.listdir(os.path.join(path, 'runs/')):
        os.remove(os.path.join(os.path.join(path, 'runs/'), directory))
    os.rmdir(os.path.join(path, 'runs/'))


def clean_test(path):
    for directory in os.listdir(os.path.join(path, 'figures/min_covers/')):
        os.remove(os.path.join(os.path.join(path, 'figures/min_covers/'), directory))
    for directory in os.listdir(os.path.join(path, 'figures/approx_min_covers/')):
        os.remove(os.path.join(os.path.join(path, 'figures/approx_min_covers/'), directory))
    for directory in os.listdir(os.path.join(path, 'figures/with_supervised_learning/')):
        os.remove(os.path.join(os.path.join(path, 'figures/with_supervised_learning/'), directory))
    for directory in os.listdir(os.path.join(path, 'figures/with_supervised_learning_1/')):
        os.remove(os.path.join(os.path.join(path, 'figures/with_supervised_learning_1/'), directory))
    for directory in os.listdir(os.path.join(path, 'figures/with_supervised_learning_2/')):
        os.remove(os.path.join(os.path.join(path, 'figures/with_supervised_learning_2/'), directory))
    for directory in os.listdir(os.path.join(path, 'figures/boxes/')):
        os.remove(os.path.join(os.path.join(path, 'figures/boxes/'), directory))
    os.rmdir(os.path.join(path, 'figures/min_covers/'))
    os.rmdir(os.path.join(path, 'figures/approx_min_covers/'))
    os.rmdir(os.path.join(path, 'figures/with_supervised_learning/'))
    os.rmdir(os.path.join(path, 'figures/with_supervised_learning_1/'))
    os.rmdir(os.path.join(path, 'figures/with_supervised_learning_2/'))
    os.rmdir(os.path.join(path, 'figures/boxes/'))
    os.rmdir(os.path.join(path, 'figures/'))


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--clean', action='store_true')
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    clean = arguments.clean
    path = os.path.join('./runs/', arguments.path)
    main(clean, path)
