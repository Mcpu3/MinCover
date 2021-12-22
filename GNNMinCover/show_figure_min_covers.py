from argparse import ArgumentParser
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main(graph_id, path):
    figure = plt.figure(constrained_layout=True)
    figure.suptitle('Graph id: {}'.format(graph_id))
    images, titles = [], []
    try:
        images.append(mpimg.imread(os.path.join(path, 'figures/min_covers/{}.jpg'.format(graph_id))))
        titles.append('Min cover')
    except:
        pass
    try:
        images.append(mpimg.imread(os.path.join(path, 'figures/approx_min_covers/{}.jpg'.format(graph_id))))
        titles.append('Approx min cover')
    except:
        pass
    try:
        images.append(mpimg.imread(os.path.join(path, 'figures/with_supervised_learning/{}.jpg'.format(graph_id))))
        titles.append('With supervised learning')
    except:
        pass
    try:
        images.append(mpimg.imread(os.path.join(path, 'figures/with_supervised_learning_1/{}.jpg'.format(graph_id))))
        titles.append('With supervised learning 1')
    except:
        pass
    try:
        images.append(mpimg.imread(os.path.join(path, 'figures/with_supervised_learning_2/{}.jpg'.format(graph_id))))
        titles.append('With supervised learning 2')
    except:
        pass
    for index, (image, title) in enumerate(zip(images, titles)):
        axis = figure.add_subplot(1, len(images), index + 1)
        axis.set_axis_off()
        axis.set_title(title)
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--graph_id', type=int, required=True)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    graph_id = arguments.graph_id
    path = os.path.join('./runs/', arguments.path)
    main(graph_id, path)
