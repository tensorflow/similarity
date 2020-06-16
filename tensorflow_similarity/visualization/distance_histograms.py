import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors
from tensorflow_similarity.visualization.utils import filter_data
from matplotlib.ticker import PercentFormatter


def plot_distance_histograms(
        x_test,
        y_test,
        x_targets,
        y_targets,
        title="Distance Histograms",
        classes=None):
    """Plot the distance histogram given the test and target data.

    Args:
        x_test (List[Embeddings]): The list of embeddings of test dataset.
        y_test (List[Integers]): The list of labels of test dataset.
        x_targets (List[Embeddings]): The list of embeddings of target dataset.
        y_targets (List[Integers]): The list of labels of target dataset.
        title (str, optional): Title of figure. Defaults to "Nearest Neighbors".
        classes (Array[String|Integers]): The array-like parameter that specify
            which classes to display, when None then we display all classes.
            Defaults to None.

    Returns:
        figure (matplotlib figure): The figure object that contains the
            distance historgram.
    """

    # filter test and targets data if classes is specified
    if classes is not None:
        x_test, y_test, _ = filter_data(
            x_test, y_test, classes)
        x_targets, y_targets, _ = filter_data(
            x_targets, y_targets, classes)

    # compute nearest neighbors for each of the targets
    neighbors_database = NearestNeighbors(
        n_neighbors=len(y_targets)).fit(x_targets)
    distances, indices = neighbors_database.kneighbors(x_test)

    # dictionary of dictionary where the key of the outer dictionary
    # is the label and the key to the inner dictionary is either 'correct'
    # or 'incorrect', where 'correct' means the distances to the closest
    # correct target for each samples of that label, and 'incorrect' means
    # the distances to the closest incorrect target for each samples of that
    # label
    distances_storage = defaultdict(lambda: defaultdict(list))
    all_samples_key = "All Samples"

    for i, true_label in enumerate(y_test):
        neighbor = indices[i]
        closest_correct_index = [y_targets[i]
                                 for i in neighbor].index(true_label)
        # closest incorrect index is 0 if closest_correect_index is not 0
        closest_incorrect_index = 0 if closest_correct_index else 1

        incorrect_distance = distances[i][closest_incorrect_index]
        correct_distance = distances[i][closest_correct_index]
        distances_storage[str(true_label)]["correct"].append(correct_distance)
        distances_storage[str(true_label)]["incorrect"].append(
            incorrect_distance)
        distances_storage[all_samples_key]["correct"].append(correct_distance)
        distances_storage[all_samples_key]["incorrect"].append(
            incorrect_distance)

    num_classes = len(distances_storage)

    # due to fix width of window size on colab and notebook, we want to
    # limit the number of columns to 3
    num_cols = min(num_classes, 3)
    # one additional row for legends
    num_rows = math.ceil(num_classes / num_cols) + 1

    # add a row for legend if all subplot will be occupied
    if num_cols * num_rows == len(distances_storage):
        num_rows += 1

    width = 20
    # The number of rows in this visualization is dynamic based on
    # number of classes, therefore height of the plot is scaled with number
    # of classes
    height = max(num_classes * 2, 20)

    figure, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, height), sharex=True, sharey=True)
    figure.suptitle(title, fontsize=30)

    for i, (key, value) in enumerate(sorted(distances_storage.items())):

        row_id = i // num_cols
        col_id = i % num_cols
        ax = axes[row_id][col_id]

        # weights to normalize y-axis for each subplot
        weights = np.ones(len(value["correct"])) / len(value["correct"])

        # use different coor for "All Samples" subplot to indicate that this
        # subplot contains aggregated data
        if key == all_samples_key:
            ax.hist(value["correct"], alpha=0.6, color="green", weights=weights)
            ax.hist(value["incorrect"], alpha=0.8, color="red", weights=weights)
        else:
            # use blue and orange for accessibility (color-blindness)
            ax.hist(value["correct"], alpha=0.6, color="blue", weights=weights)
            ax.hist(
                value["incorrect"],
                alpha=0.8,
                color="orange",
                weights=weights)

        # set the title of the subplot to the label of the test data points
        ax.set_title("Class: {}".format(key), fontsize=25)

        # set the y-axis from decimenal to percentage (0.23 -> 23%)
        ax.yaxis.set_major_formatter(PercentFormatter(1))

        # plot the vertical average line for easier comparsion
        mean_correct = np.mean(value["correct"])
        mean_incorrect = np.mean(value["incorrect"])
        ax.axvline(mean_correct, color='black', linestyle='solid', linewidth=2)
        ax.axvline(
            mean_incorrect,
            color='black',
            linestyle='dashed',
            linewidth=2)

        # first column, add y labels
        if col_id == 0:
            ax.set_ylabel("number of samples", fontsize=20)
        # last row, add x labels and turn the x ticks back on
        if row_id == num_rows - 2:
            ax.set_xlabel("distance", fontsize=20)
            ax.xaxis.set_tick_params(labelbottom=True)

    # remove empty plots in the excess subplots
    for excess_index in range(len(distances_storage), num_cols * num_rows):
        axes.flat[excess_index].set_axis_off()

    # create a legend as it's own subplots (last row of the figure)
    grid_spec = axes[0, 0].get_gridspec()
    ax = figure.add_subplot(grid_spec[-1, 0:])

    # create custom handlers for the legend
    correct_targets_handle = tuple(
        [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ["blue", "green"]])
    incorrect_targets_handle = tuple(
        [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ["orange", "red"]])
    correct_mean_handle = Line2D(
        [0],
        [0],
        color='black',
        linestyle='solid',
        linewidth=2)
    incorrect_mean_handle = Line2D(
        [0],
        [0],
        color='black',
        linestyle='dashed',
        linewidth=2)
    handles = [
        correct_targets_handle,
        incorrect_targets_handle,
        correct_mean_handle,
        incorrect_mean_handle]

    # labels for each handler in the legend
    labels = ["distances to closest correct target",
              "distances to closest incorrect target",
              "mean distance to closest correct target",
              "mean distance to closest incorrect target"]

    ax.axis('off')
    ax.legend(
        handles, labels, loc='upper left', prop={
            'size': 20}, ncol=2, handler_map={
            tuple: HandlerTuple(
                ndivide=None)})

    # preventing overlaps
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)

    return figure
