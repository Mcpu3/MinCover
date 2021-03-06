from argparse import ArgumentParser
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(without_min_covers, without_min_covers_approx, without_with_supervised_learning, without_with_supervised_learning_1, without_with_supervised_learning_2, path, file_name):
    event_accumulator = EventAccumulator(os.path.join(os.path.join(path, 'runs/'), file_name)).Reload()
    graphs_id, accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2, columns = [], [], [], [], [], [], [], []
    if not without_min_covers:
        accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers, evaluations_1_of_min_covers, evaluations_2_of_min_covers = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestMinCover')
        graphs_id_of_min_covers, accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers, evaluations_1_of_min_covers, evaluations_2_of_min_covers = from_list_of_scalar_event_to_list(accs_of_min_covers, aucs_of_min_covers, aps_of_min_covers, times_elapsed_of_min_covers, evaluations_1_of_min_covers, evaluations_2_of_min_covers)
        graphs_id = graphs_id_of_min_covers
        accs.append(accs_of_min_covers)
        aucs.append(aucs_of_min_covers)
        aps.append(aps_of_min_covers)
        times_elapsed.append(times_elapsed_of_min_covers)
        evaluations_1.append(evaluations_1_of_min_covers)
        evaluations_2.append(evaluations_2_of_min_covers)
        columns.append('Min covers')
    if not without_min_covers_approx:
        accs_of_min_covers_approx, aucs_of_min_covers_approx, aps_of_min_covers_approx, times_elapsed_of_min_covers_approx, evaluations_1_of_min_covers_approx, evaluations_2_of_min_covers_approx = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestApproxMinCover')
        graphs_id_of_min_covers_approx, accs_of_min_covers_approx, aucs_of_min_covers_approx, aps_of_min_covers_approx, times_elapsed_of_min_covers_approx, evaluations_1_of_min_covers_approx, evaluations_2_of_min_covers_approx = from_list_of_scalar_event_to_list(accs_of_min_covers_approx, aucs_of_min_covers_approx, aps_of_min_covers_approx, times_elapsed_of_min_covers_approx, evaluations_1_of_min_covers_approx, evaluations_2_of_min_covers_approx)
        graphs_id = graphs_id_of_min_covers_approx
        accs.append(accs_of_min_covers_approx)
        aucs.append(aucs_of_min_covers_approx)
        aps.append(aps_of_min_covers_approx)
        times_elapsed.append(times_elapsed_of_min_covers_approx)
        evaluations_1.append(evaluations_1_of_min_covers_approx)
        evaluations_2.append(evaluations_2_of_min_covers_approx)
        columns.append('Approx min covers')
    if not without_with_supervised_learning:
        accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning, evaluations_1_of_with_supervised_learning, evaluations_2_of_with_supervised_learning = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestWithSupervisedLearning')
        graphs_id_of_with_supervised_learning, accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning, evaluations_1_of_with_supervised_learning, evaluations_2_of_with_supervised_learning = from_list_of_scalar_event_to_list(accs_of_with_supervised_learning, aucs_of_with_supervised_learning, aps_of_with_supervised_learning, times_elapsed_of_with_supervised_learning, evaluations_1_of_with_supervised_learning, evaluations_2_of_with_supervised_learning)
        graphs_id = graphs_id_of_with_supervised_learning
        accs.append(accs_of_with_supervised_learning)
        aucs.append(aucs_of_with_supervised_learning)
        aps.append(aps_of_with_supervised_learning)
        times_elapsed.append(times_elapsed_of_with_supervised_learning)
        evaluations_1.append(evaluations_1_of_with_supervised_learning)
        evaluations_2.append(evaluations_2_of_with_supervised_learning)
        columns.append('With supervised learning')
    if not without_with_supervised_learning_1:
        accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1, evaluations_1_of_with_supervised_learning_1, evaluations_2_of_with_supervised_learning_1 = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestWithSupervisedLearning1')
        graphs_id_of_with_supervised_learning_1, accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1, evaluations_1_of_with_supervised_learning_1, evaluations_2_of_with_supervised_learning_1 = from_list_of_scalar_event_to_list(accs_of_with_supervised_learning_1, aucs_of_with_supervised_learning_1, aps_of_with_supervised_learning_1, times_elapsed_of_with_supervised_learning_1, evaluations_1_of_with_supervised_learning_1, evaluations_2_of_with_supervised_learning_1)
        graphs_id = graphs_id_of_with_supervised_learning_1
        accs.append(accs_of_with_supervised_learning_1)
        aucs.append(aucs_of_with_supervised_learning_1)
        aps.append(aps_of_with_supervised_learning_1)
        times_elapsed.append(times_elapsed_of_with_supervised_learning_1)
        evaluations_1.append(evaluations_1_of_with_supervised_learning_1)
        evaluations_2.append(evaluations_2_of_with_supervised_learning_1)
        columns.append('With supervised learning 1')
    if not without_with_supervised_learning_2:
        accs_of_with_supervised_learning_2, aucs_of_with_supervised_learning_2, aps_of_with_supervised_learning_2, times_elapsed_of_with_supervised_learning_2, evaluations_1_of_with_supervised_learning_2, evaluations_2_of_with_supervised_learning_2 = from_event_accumulator_to_list_of_scalar_event(event_accumulator, 'TestWithSupervisedLearning2')
        graphs_id_of_with_supervised_learning_2, accs_of_with_supervised_learning_2, aucs_of_with_supervised_learning_2, aps_of_with_supervised_learning_2, times_elapsed_of_with_supervised_learning_2, evaluations_1_of_with_supervised_learning_2, evaluations_2_of_with_supervised_learning_2 = from_list_of_scalar_event_to_list(accs_of_with_supervised_learning_2, aucs_of_with_supervised_learning_2, aps_of_with_supervised_learning_2, times_elapsed_of_with_supervised_learning_2, evaluations_1_of_with_supervised_learning_2, evaluations_2_of_with_supervised_learning_2)
        graphs_id = graphs_id_of_with_supervised_learning_2
        accs.append(accs_of_with_supervised_learning_2)
        aucs.append(aucs_of_with_supervised_learning_2)
        aps.append(aps_of_with_supervised_learning_2)
        times_elapsed.append(times_elapsed_of_with_supervised_learning_2)
        evaluations_1.append(evaluations_1_of_with_supervised_learning_2)
        evaluations_2.append(evaluations_2_of_with_supervised_learning_2)
        columns.append('With supervised learning 2')
    data_of_accs = pd.DataFrame(np.transpose(np.array(accs)), graphs_id, columns)
    data_of_aucs = pd.DataFrame(np.transpose(np.array(aucs)), graphs_id, columns)
    data_of_aps = pd.DataFrame(np.transpose(np.array(aps)), graphs_id, columns)
    data_of_times_elapsed = pd.DataFrame(np.transpose(np.array(times_elapsed)), graphs_id, columns)
    data_of_evaluations_1 = pd.DataFrame(np.transpose(np.array(evaluations_1)), graphs_id, columns)
    data_of_evaluations_2 = pd.DataFrame(np.transpose(np.array(evaluations_2)), graphs_id, columns)
    save_figure(data_of_accs, False, 'Acc', os.path.join(path, 'figures/boxes/accs.jpg'))
    save_figure(data_of_aucs, False, 'AUC', os.path.join(path, 'figures/boxes/aucs.jpg'))
    save_figure(data_of_aps, False, 'AP', os.path.join(path, 'figures/boxes/aps.jpg'))
    save_figure(data_of_times_elapsed, True, 'Elapsed time [s]', os.path.join(path, 'figures/boxes/elapsed_times.jpg'))
    save_figure(data_of_evaluations_1, False, 'Evaluation 1', os.path.join(path, 'figures/boxes/evaluations_1.jpg'))
    save_figure(data_of_evaluations_2, False, 'Evaluation 2', os.path.join(path, 'figures/boxes/evaluations_2.jpg'))


def from_event_accumulator_to_list_of_scalar_event(event_accumulator, tag):
    accs = event_accumulator.Scalars('Acc/' + tag)
    aucs = event_accumulator.Scalars('AUC/' + tag)
    aps = event_accumulator.Scalars('AP/' + tag)
    times_elapsed = event_accumulator.Scalars('ElapsedTime/' + tag)
    evaluations_1 = event_accumulator.Scalars('Evaluation1/' + tag)
    evaluations_2 = event_accumulator.Scalars('Evaluation2/' + tag)
    return accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2


def from_list_of_scalar_event_to_list(accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2):
    accs_copy, aucs_copy, aps_copy, times_elapsed_copy, evaluations_1_copy, evaluations_2_copy = copy.deepcopy(accs), copy.deepcopy(aucs), copy.deepcopy(aps), copy.deepcopy(times_elapsed), copy.deepcopy(evaluations_1), copy.deepcopy(evaluations_2)
    graphs_id, accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2 = [], [], [], [], [], [], []
    for acc, auc, ap, time_elapsed, evaluation_1, evaluation_2 in zip(accs_copy, aucs_copy, aps_copy, times_elapsed_copy, evaluations_1_copy, evaluations_2_copy):
        graphs_id.append(time_elapsed.step)
        accs.append(acc.value)
        aucs.append(auc.value)
        aps.append(ap.value)
        times_elapsed.append(time_elapsed.value)
        evaluations_1.append(evaluation_1.value)
        evaluations_2.append(evaluation_2.value)
    return graphs_id, accs, aucs, aps, times_elapsed, evaluations_1, evaluations_2


def save_figure(data, is_log, x_label, path):
    axis = sns.boxplot(data=data, orient='h', showfliers=False, boxprops={'facecolor': '#fff'})
    if is_log:
        axis.set_xscale('log')
    axis.set_xlabel(x_label)
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
    argument_parser.add_argument('--without_with_supervised_learning_2', action='store_true')
    argument_parser.add_argument('--path', required=True)
    argument_parser.add_argument('--file_name', required=True)
    arguments = argument_parser.parse_args()
    without_min_covers = arguments.without_min_covers
    without_min_covers_approx = arguments.without_approx_min_covers
    without_with_supervised_learning = arguments.without_with_supervised_learning
    without_with_supervised_learning_1 = arguments.without_with_supervised_learning_1
    without_with_supervised_learning_2 = arguments.without_with_supervised_learning_2
    path = os.path.join('./runs/', arguments.path)
    file_name = arguments.file_name
    main(without_min_covers, without_min_covers_approx, without_with_supervised_learning, without_with_supervised_learning_1, without_with_supervised_learning_2, path, file_name)
