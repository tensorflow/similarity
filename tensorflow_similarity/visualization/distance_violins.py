# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from altair import datum
from sklearn.neighbors import NearestNeighbors
from tensorflow_similarity.visualization.utils import filter_data


def plot_distance_violins(
        x_test,
        y_test,
        x_targets,
        y_targets,
        title="Violins Plot",
        interactive=True,
        classes=None):
    """Plot the violin plots given the test and target data.

    Args:
        x_test (List[Embeddings]): The list of embeddings of test dataset.
        y_test (List[Integers]): The list of labels of test dataset.
        x_targets (List[Embeddings]): The list of embeddings of target dataset.
        y_targets (List[Integers]): The list of labels of target dataset.
        title (str, optional): Title of figure. Defaults to "Violins Plot".
        interactive (bool, optional): When set to true, we will return an
            interactive plot using Altair, otherwise return a static plot.
            Defaults to True.
        classes (Array[String|Integers]): The array-like parameter that specify
            which classes to display, when None then we display all classes.
            Defaults to None.

    Returns:
        figure (matplotlib figure | Altair chart): The figure object that
            contains the violins plot.
    """

    # filter test and targets data if classes is specified
    if classes is not None:
        x_test, y_test, _ = filter_data(
            x_test, y_test, classes)
        x_targets, y_targets, _ = filter_data(
            x_targets, y_targets, classes)

    # Altair breaks when we have numpy.int* object, therefore we convert
    # them to Strings for compatibility
    y_test = np.array([str(label) for label in y_test])
    y_targets = np.array([str(label) for label in y_targets])
    unique_labels = set(y_targets)

    # compute nearest neighbors for each of the targets
    neighbors_database = NearestNeighbors(
        n_neighbors=len(y_targets)).fit(x_targets)
    distances, indices = neighbors_database.kneighbors(x_test)

    distances_storage = defaultdict(list)

    for i, true_label in enumerate(y_test):
        neighbor = indices[i]
        target_indices = [y_targets[i] for i in neighbor]

        # store the closest distance to that label from the test point
        for label in unique_labels:
            closest_index = target_indices.index(label)
            distance = distances[i][closest_index]

            distances_storage["label"].append(true_label)
            distances_storage["target_label"].append(label)
            distances_storage["distance"].append(distance)

    distance_df = pd.DataFrame(distances_storage)

    # plot using Altair if user wants an interactive plot
    if interactive:
        # create the dropdown menu
        options = sorted(list(unique_labels))
        default_label = options[0]
        dropdown = alt.binding_select(options=options)
        selection = alt.selection_single(
            fields=['label'],
            bind=dropdown,
            name='select',
            init={"label": default_label})

        # set the color of the interactive violin plots
        # we will use blue (for distance distribution to correct target label)
        # and orange (for distance distribution to other target labels)
        color = alt.condition(
            "datum.target_label == select.label",
            alt.value('blue'),
            alt.value('orange'),
            legend=None)

        # create the interactive violin plots, followed the examples shown
        # here: https://altair-viz.github.io/gallery/violine_plot.html
        figure = alt.Chart(distance_df).transform_filter(
            selection
        ).transform_bin(
            ['bin_max', 'bin_min'],
            field='distance',
            bin=alt.Bin(maxbins=20)
        ).transform_calculate(
            binned=(datum.bin_max + datum.bin_min) / 2
        ).transform_aggregate(
            value_count='count()',
            groupby=['target_label', 'binned']
        ).transform_impute(
            impute='value_count',
            groupby=['target_label'],
            key='binned',
            value=0
        ).mark_area(
            interpolate='monotone',
            orient='horizontal'
        ).encode(
            x=alt.X(
                'value_count:Q',
                title=None,
                stack='center',
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            y=alt.Y('binned:Q', bin='binned', title='distance'),
            color=color,
            column=alt.Column(
                'target_label:N',
                header=alt.Header(
                    titleOrient='bottom',
                    labelOrient='bottom',
                    labelPadding=0,
                ),
            ),
        ).properties(
            width=80,
            title=title
        ).add_selection(
            selection
        ).configure_facet(
            spacing=0
        ).configure_view(
            stroke=None
        )
    else:
        # plot static plot using matplotlib and seaborn
        num_classes = len(unique_labels)

        # Each row will represent a class and the column will be formatted
        # automatically by seaborn
        num_cols = 1
        num_rows = num_classes

        width = 20
        # The number of rows in this visualization is dynamic based on
        # number of classes, therefore height of the plot is scaled with number
        # of classes
        height = max(num_classes * 6, 10)

        figure, axes = plt.subplots(
            num_rows, num_cols, figsize=(width, height), sharey=True)
        figure.suptitle(title, fontsize=30)

        for i, label in enumerate(sorted(unique_labels)):

            if isinstance(axes, np.ndarray):
                ax = axes[i]
            else:
                ax = axes

            # set the title of the subplot to the label of the test data points
            ax.set_title("Class: {}".format(label), fontsize=25)

            # blue for distance distribution to the correct target and
            # orange for distance distribution for any other target for strong
            # and colorblind-proof contrast
            palette = defaultdict(lambda: "orange")
            palette[label] = "blue"

            data = distance_df[distance_df["label"] == label]

            # plot the violin plot on this subplot
            sns.violinplot(x="target_label",
                           y="distance",
                           data=data,
                           palette=palette,
                           ax=ax)

            # set the size for x and y label for readability
            ax.xaxis.label.set_size(20)
            ax.set_xlabel("target class")
            ax.yaxis.label.set_size(20)
            ax.tick_params(labelsize=20)

            # compute the mean and standard deviation of distances to the
            # correct target
            correct_distances = data[data["target_label"] == label]["distance"]
            correct_mean = np.mean(correct_distances)
            correct_std = np.std(correct_distances)

            # draw two horizontal lines in each subplot, one for the mean
            # distance to the correct target and another for the mean + 2*std
            # distance to the correct target, those two lines will make it
            # easier for users to visualize which other target that the model
            # does not cluster well.
            ax.axhline(correct_mean, color="black", linestyle="dashed")
            ax.axhline(
                correct_mean + 2 * correct_std,
                color="black",
                linestyle="dashed")

        # preventing overlaps
        figure.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)

    return figure
