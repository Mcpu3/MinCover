from argparse import ArgumentParser
import os

import seaborn as sns
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(path, file_name):
    print(os.path.join(os.path.join(path, 'runs/'), file_name))
    event_accumulator = EventAccumulator(os.path.join(os.path.join(path, 'runs/'), file_name))
    event_accumulator.Reload()
    accs_of_min_cover = event_accumulator.Scalars('Acc/TestMinCover')
    aucs_of_min_cover = event_accumulator.Scalars('AUC/TestMinCover')
    aps_of_min_cover = event_accumulator.Scalars('AP/TestMinCover')
    accs_of_min_cover_approx = event_accumulator.Scalars('Acc/TestApproxMinCover')
    aucs_of_min_cover_approx = event_accumulator.Scalars('AUC/TestApproxMinCover')
    aps_of_min_cover_approx = event_accumulator.Scalars('AP/TestApproxMinCover')
    accs_of_min_cover_with_supervised_learning = event_accumulator.Scalars('Acc/TestMinCoverWithSupervisedLearning')
    aucs_of_min_cover_with_supervised_learning = event_accumulator.Scalars('AUC/TestMinCoverWithSupervisedLearning')
    aps_of_min_cover_with_supervised_learning = event_accumulator.Scalars('AP/TestMinCoverWithSupervisedLearning')
    x = []
    data = [[], [], []]
    for index, (acc, auc, ap) in enumerate(zip(accs_of_min_cover, aucs_of_min_cover, aps_of_min_cover)):
        x.append(index)
        data[0].append(acc)
        data[1].append(auc)
        data[2].append(ap)
    df = pd.DataFrame(data, columns=['accs', 'aucs', 'aps'])
    print(df)


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--path', required=True)
    argument_parser.add_argument('--file_name', required=True)
    arguments = argument_parser.parse_args()
    path = os.path.join('./runs/', arguments.path)
    file_name = arguments.file_name
    main(path, file_name)
