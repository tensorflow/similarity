import itertools

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from tensorflow_similarity.visualization.utils import filter_data


def plot_confusion_matrix(
        x_test,
        y_test,
        x_targets,
        y_targets,
        title="Confusion Matrix",
        classes=None):
    """Plot the confusion matrix given the test and target data.

    Args:
        x_test (List[Embeddings]): The list of embeddings of test dataset.
        y_test (List[Integers]): The list of labels of test dataset.
        x_targets (List[Embeddings]): The list of embeddings of target dataset.
        y_targets (List[Integers]): The list of labels of target dataset.
        title (str, optional): Title of figure. Defaults to "Confusion Matrix".
        classes (Array[String|Integers]): The array-like parameter that specify
            which classes to display, when None then we display all classes.
            Defaults to None.

    Returns:
        figure (matplotlib figure): The figure object that contains the
            confusion matrix.
    """

    # filter test and targets data if classes is specified
    if classes is not None:
        x_test, y_test, _ = filter_data(x_test, y_test, classes)
        x_targets, y_targets, _ = filter_data(x_targets, y_targets, classes)

    distances = sklearn.metrics.pairwise.euclidean_distances(x_test, x_targets)
    y_pred_indices = np.argmin(distances, axis=1)
    y_pred = np.array([y_targets[index] for index in y_pred_indices])

    # In N-way k-shot learning we will have k targets per class, we
    # just need the unique ones, this simple approach also preserves order.
    unique_target_labels = list(set(y_targets))
    num_classes = len(unique_target_labels)

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_test, y_pred, unique_target_labels)

    # scale the size of the plot by the number of classes
    size = max(num_classes, 8)

    figure = plt.figure(figsize=(size, size))
    plt.imshow(
        confusion_matrix,
        interpolation='nearest',
        cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, unique_target_labels, rotation=45)
    plt.yticks(tick_marks, unique_target_labels)

    # Normalize the confusion matrix.
    epsilon = 10 ** -10
    row_sum = confusion_matrix.sum(axis=1)[:, np.newaxis] + epsilon
    confusion_matrix = np.around(
        confusion_matrix.astype('float') / row_sum, decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = confusion_matrix.max() / 2.0
    for i, j in itertools.product(
        range(
            confusion_matrix.shape[0]), range(
            confusion_matrix.shape[1])):
        color = "white" if confusion_matrix[i, j] > threshold else "black"
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)

    return figure
