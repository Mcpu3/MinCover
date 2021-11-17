from argparse import ArgumentParser
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main(g_id):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Graph Id: {}'.format(g_id))
    imges = []
    titles = []
    try:
        imges.append(mpimg.imread('./fig/min_covers/{}.jpg'.format(g_id)))
        titles.append('Min Cover')
    except:
        pass
    try:
        imges.append(mpimg.imread(
            './fig/approx_min_covers/{}.jpg'.format(g_id)))
        titles.append('Approx Min Cover')
    except:
        pass
    try:
        imges.append(mpimg.imread('./fig/tests/{}.jpg'.format(g_id)))
        titles.append('Test')
    except:
        pass
    for ind, img, title in zip(range(len(imges)), imges, titles):
        ax = fig.add_subplot(1, len(imges), ind + 1)
        ax.set_axis_off()
        ax.set_title(title)
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--g_id', type=int)
    args = parser.parse_args()
    main(args.g_id)
