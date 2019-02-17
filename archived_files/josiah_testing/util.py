#!/usr/bin/env python
import itertools
import os
from pathlib import Path
from typing import List
import numpy as np
import warnings
import matplotlib.pyplot as plt


def get_absolute_data_path(data_workspace: str = 'data', parent_up_limit=2, sub_folder_name: str = None):
    """
    Gets the absolute path for the 'data' directory.

    ***Note that this assumes the script is in the lowes-product-classifier***

    :param sub_folder_name: A sub folder name. It is the user's responsibility if the sub folder has separators such as
    '/' or '\'. Please use os.sep if this is the case.
    :param data_workspace: Name of the workspace. Default is 'data'
    :param parent_up_limit: The number of upper folders to look through to find the directory
    :return: The absolute path to the workspace. IE a string like:
    /Users/jlaivins/PycharmProjects/Lowes-Product-Classifier/lowes-product-classifier/data/
    """
    absolute_path = ''
    for i in range(-1, parent_up_limit):
        if i == -1:
            curr = str(Path().absolute())
        else:
            curr = str(Path().absolute().parents[i])
        if data_workspace in os.listdir(curr):
            absolute_path = curr + os.sep + data_workspace + os.sep
            break

    # If the user specifies a sub folder, add it
    if sub_folder_name is not None:
        absolute_path += sub_folder_name

    return absolute_path


def rename_filenames(files: List[str], data_workspace: str = 'data', prefix: str = 'im', postfix: str = '',
                     regex: str = '.'):
    # Get the absolute path to the data folder
    absolute_data_workspace_path = get_absolute_data_path(data_workspace)

    for file in files:
        # Get the file names in the directory
        document_names = os.listdir(absolute_data_workspace_path + file)
        increment = 0
        # Go through each image
        for document in document_names:
            full_path = absolute_data_workspace_path + file + os.sep + document
            # If the path exists
            if os.path.exists(full_path):
                rename(full_path, document, prefix, postfix, regex, increment)
            else:
                warnings.warn(f'File: {document} was not found in {full_path}. Verify that this file exists.',
                              RuntimeWarning)
            increment += 1


def rename(absolute_document_path: str, original_name: str, prefix: str, postfix: str, regex: str, inc: int = 0):
    # Get the new absolute path
    new_absolute_document_path = absolute_document_path[:-len(original_name)] + prefix + postfix + str(inc) + \
                                 original_name[original_name.index(regex):]

    print(f'Changing {absolute_document_path} to {new_absolute_document_path}')
    if not os.path.exists(new_absolute_document_path):
        os.rename(absolute_document_path, new_absolute_document_path)
    else:
        warnings.warn(f'The new file path: {new_absolute_document_path} already exists. Skipping renaming. ')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    classes = [c[:str(c).index('_', int(len(c)/4))] for c in classes]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_precision_recall_f1(metrics, metric_labels,
                             title='Metrics for: ',
                             ):
    import matplotlib.pyplot as plt
    import itertools
    cmap =plt.cm.Blues
    metrics = np.array(metrics[:-1]).reshape(-1, 1).T

    plt.imshow(metrics, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    tick_marks = np.arange(len(metric_labels))
    plt.xticks(tick_marks, metric_labels, rotation=45)
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False)

    fmt = '.2f'
    thresh = np.average(metrics)
    for i, j in itertools.product(range(metrics.shape[0]), range(metrics.shape[1])):
        plt.text(j, i, format(metrics[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if metrics[i, j] > thresh else "black")

    plt.tight_layout()


if __name__ == '__main__':
    # rename_filenames(['candle_holder', 'screw_sheet_metal_1in', 'screw_socket_1in'])
    get_absolute_data_path()