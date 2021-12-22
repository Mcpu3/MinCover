from argparse import ArgumentParser
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(without_min_covers, without_min_covers_approx, without_with_supervised_learning, without_with_supervised_learning_1, path, file_name):
    event_accumulator = EventAccumulator(os.path.join(os.path.join(path, 'runs/'), file_name)).Reload()
    graphs_id, accs, aucs, aps, times_elapsed, columns = [], [], [], [], [], []
    if not without_min_covers:
        accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestMinCover')
        graphs_id_of_min_covers, accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers = from_list_of_scalar_event_to_list(accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers)
        graphs_id = graphs_id_of_min_covers
        accs.append(accs_of_min_covers)
        aucs.append(aucs_of_min_covers)
        aps.append(aps_of_min_covers)
        times_elapsed.append(times_elapsed_of_min_covers)
        columns.append('Min covers')
    if not without_min_covers_approx:
        accs_of_min_covers_approx, aucs_of_min_covers_approx, aps_of_min_covers_approx, times_elapsed_of_min_covers_approx = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestApproxMinCover')
        graphs_id_of_min_covers_approx, accs_of_min_covers_approx, aucs_of_min_covers_approx, aps_of_min_covers_approx, times_elapsed_of_min_covers_approx = from_list_of_scalar_event_to_list(accs_of_min_covers_approx, aucs_of_min_covers_approx, aps_of_min_covers_approx, times_elapsed_of_min_covers_approx)
        graphs_id = graphs_id_of_min_covers_approx
        accs.append(accs_of_min_covers_approx)
        aucs.append(aucs_of_min_covers_approx)
        aps.append(aps_of_min_covers_approx)
        times_elapsed.append(times_elapsed_of_min_covers_approx)
        columns.append('Approx min covers')
    if not without_with_supervised_learning:
        accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestWithSupervisedLearning')
        graphs_id_of_with_supervised_learning, accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning = from_list_of_scalar_event_to_list(accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning)
        graphs_id = graphs_id_of_with_supervised_learning
        accs.append(accs_of_with_supervised_learning)
        aucs.append(aucs_of_with_supervised_learning)
        aps.append(aps_of_with_supervised_learning)
        times_elapsed.append(times_elapsed_of_with_supervised_learning)
        columns.append('With supervised learning')
    if not without_with_supervised_learning_1:
        accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1 = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestWithSupervisedLearning1')
        graphs_id_of_with_supervised_learning_1, accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1 = from_list_of_scalar_event_to_list(accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1)
        graphs_id = graphs_id_of_with_supervised_learning_1
        accs.append(accs_of_with_supervised_learning_1)
        aucs.append(aucs_of_with_supervised_learning_1)
        aps.append(aps_of_with_supervised_learning_1)
        times_elapsed.append(times_elapsed_of_with_supervised_learning_1)
        columns.append('With supervised learning 1')
    data_of_accs = pd.DataFrame(np.transpose(np.array(accs)), graphs_id, columns)
    data_of_aucs = pd.DataFrame(np.transpose(np.array(aucs)), graphs_id, columns)
    data_of_aps = pd.DataFrame(np.transpose(np.array(aps)), graphs_id, columns)
    data_of_times_elapsed = pd.DataFrame(np.transpose(np.array(times_elapsed)), graphs_id, columns)
    save_figure(data_of_accs, False, 'Acc', None, os.path.join(path, 'figures/boxes/accs.jpg'))
    save_figure(data_of_aucs, False, 'AUC', None, os.path.join(path, 'figures/boxes/aucs.jpg'))
    save_figure(data_of_aps, False, 'AP', None, os.path.join(path, 'figures/boxes/aps.jpg'))
    save_figure(data_of_times_elapsed, True, 'Elapsed time', '[s]', os.path.join(path, 'figures/boxes/elapsed_times.jpg'))


def from_event_accumulator_to_list_of_scalar_event(event_accumulator, tag):
    accs = event_accumulator.Scalars('Acc/' + tag)
    aucs = event_accumulator.Scalars('AUC/' + tag)
    aps = event_accumulator.Scalars('AP/' + tag)
    times_elapsed = event_accumulator.Scalars('ElapsedTime/' + tag)
    return accs, aucs, aps, times_elapsed


def from_list_of_scalar_event_to_list(accs, aucs, aps, times_elapsed):
    accs_copy, aucs_copy, aps_copy, times_elapsed_copy = copy.deepcopy(accs), copy.deepcopy(aucs), copy.deepcopy(aps), copy.deepcopy(times_elapsed)
    graphs_id, accs, aucs, aps, times_elapsed = [], [], [], [], []
    for acc, auc, ap, time_elapsed in zip(accs_copy, aucs_copy, aps_copy, times_elapsed_copy):
        graphs_id.append(time_elapsed.step)
        accs.append(acc.value)
        aucs.append(auc.value)
        aps.append(ap.value)
        times_elapsed.append(time_elapsed.value)
    return graphs_id, accs, aucs, aps, times_elapsed


def save_figure(data, is_log, title, y_label, path):
    axis = sns.boxplot(data=data)
    if is_log:
        axis.set_yscale('log')
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.grid(True, 'both')
    plt.tight_layout()
    plt.savefig(path, dpi=300, pil_kwargs={'quality': 85})
    plt.close()


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--without_min_covers', action='store_true')
    argument_parser.add_argument('--without_approx_min_covers', action='store_true')
    argument_parser.add_argument('--without_with_supervised_learning', action='store_true')
    argument_parser.add_argument('--without_with_supervised_learning_1', action='store_true')
    argument_parser.add_argument('--path', required=True)
    argument_parser.add_argument('--file_name', required=True)
    arguments = argument_parser.parse_args()
    without_min_covers = arguments.without_min_covers
    without_min_covers_approx = arguments.without_approx_min_covers
    without_with_supervised_learning = arguments.without_with_supervised_learning
    without_with_supervised_learning_1 = arguments.without_with_supervised_learning_1
    path = os.path.join('./runs/', arguments.path)
    file_name = arguments.file_name
    main(without_min_covers, without_min_covers_approx, without_with_supervised_learning, without_with_supervised_learning_1, path, file_name)
