from argparse import ArgumentParser
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main(graph_id, path):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Graph Id: {}'.format(graph_id))
    images = []
    titles = []
    try:
        images.append(mpimg.imread(os.path.join(path, 'figures/min_covers/{}.jpg'.format(graph_id))))
        titles.append('Min Cover')
    except:
        pass
    try:
        images.append(mpimg.imread(os.path.join(path, 'figures/approx_min_covers/{}.jpg'.format(graph_id))))
        titles.append('Approx Min Cover')
    except:
        pass
    try:
        images.append(mpimg.imread(os.path.join(path, 'figures/tests/{}.jpg'.format(graph_id))))
        titles.append('Test')
    except:
        pass
    for index, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1, len(images), index + 1)
        ax.set_axis_off()
        ax.set_title(title)
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--graph_id', type=int, default=500)
    argument_parser.add_argument('--path', required=True)
    arguments = argument_parser.parse_args()
    graph_id = arguments.graph_id
    path = os.path.join('./runs/', arguments.path)
    main(graph_id, path)
