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

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tensorflow_similarity.visualization.utils import filter_data
import seaborn as sns
import numpy as np
from textwrap import wrap


def plot_nearest_neighbors(
        x_test,
        y_test,
        x_targets,
        y_targets,
        test_sprites,
        target_sprites,
        title="Nearest Neighbors",
        num_neighbors=5,
        classes=None):
    """Plot the nearest neighbors given the test and target data.

    In this visualization we are visualizing the nearest data points within
    the test dataset from each of the target data points. Since Tensorflow
    Similarity works with arbitrary data types as inputs, the sprites shown in
    this visualization will also works with multitude of data types. The users
    can give a list of images as the sprites (for example the MNIST dataset),
    or a list of tabular data as the sprites (for example the IRIS dataset),
    which we will then plot out the bar plots, or texts (for example domain
    spoofing), in which we will display the text.

    Args:
        x_test (List[Embeddings]): The list of embeddings of test dataset.
        y_test (List[Integers]): The list of labels of test dataset.
        x_targets (List[Embeddings]): The list of embeddings of target dataset.
        y_targets (List[Integers]): The list of labels of target dataset.
        test_sprites (List[Sprite]): The list of sprites (2D/3D for images,
            1D for tabular, other shape for text) for the test dataset.
        target_sprites (List[Sprite]): The list of sprites (2D/3D for images,
            1D for tabular, other shape for text) for the target dataset.
        title (str, optional): Title of figure. Defaults to "Nearest Neighbors".
        num_neighbors (int): Optional, the number of neighbors shown per
            target input. Defaults to 5.
        classes (Array[String|Integer]): The array-like parameter that specify
            which classes to display, when None then we display all classes.
            Defaults to None.

    Returns:
        figure (matplotlib figure): The figure object that contains the
            nearest neighbors.
    """

    # filter test and targets data if classes is specified
    if classes is not None:
        x_test, y_test, test_sprites = filter_data(
            x_test, y_test, classes, test_sprites)
        x_targets, y_targets, target_sprites = filter_data(
            x_targets, y_targets, classes, target_sprites)

    # sort targets for better display
    x_targets = [x for _, x in sorted(zip(y_targets, x_targets))]
    target_sprites = [sprite for _, sprite in sorted(
        zip(y_targets, target_sprites))]
    y_targets = sorted(y_targets)

    # compute nearest neighbors for each of the targets
    neighbors_database = NearestNeighbors(
        n_neighbors=num_neighbors).fit(x_test)
    distances, indices = neighbors_database.kneighbors(x_targets)

    # figure out the data type of sprite data, currently we support 3 modes:
    # images (when each sprite has dimension of 2 or 3), tabular (when
    # each sprite is an array-like object (1 dimension)), or other
    # which we will then convert into String.
    sprite_type = "text"
    first_sprite = test_sprites[0]
    if isinstance(first_sprite, np.ndarray):
        sprite_dimension = first_sprite.ndim
        # if the sprite dimension is 2 or 3 then it is an image, Video type
        # is currently not supported
        if sprite_dimension == 2 or sprite_dimension == 3:
            sprite_type = "image"
        elif sprite_dimension == 1:
            sprite_type = "tabular"

    # wrap the String sprites so that long strings wouldn't overlap with
    # other subplots
    if sprite_type == "text":
        char_per_line = 17
        target_sprites = ['\n'.join(wrap(l, char_per_line))
                          for l in target_sprites]
        test_sprites = ['\n'.join(wrap(l, char_per_line))
                        for l in test_sprites]
        target_bbox = dict(
            boxstyle='round',
            facecolor='turquoise',
            alpha=0.5)
        correct_label_bbox = dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.5)
        incorrect_label_bbox = dict(
            boxstyle='round',
            facecolor='red',
            alpha=0.5)
    elif sprite_type == "tabular":
        # for tabular data type visualizatoin we need this array to plot
        # the bar plot, since the array is the same for each subplot,
        # we will initilize it here and reuse it
        x_pos = np.arange(len(target_sprites[0]))

    # one row per target
    num_row = len(y_targets)
    # add 2 for one additonal column for target image and
    # another for divider column.
    num_col = num_neighbors + 2

    # scale the size of the plot by factor of 2 for better
    # looking display.
    scale = 2
    plt_width = num_col * scale
    plt_height = num_row * scale

    # Share both x and y axis for consistancy across subplots
    figure, axes = plt.subplots(num_row, num_col, figsize=(
        plt_width, plt_height), sharey=True, sharex=True)
    figure.suptitle(title, fontsize=24)

    for row_id in range(num_row):
        for col_id in range(num_col):
            ax = axes[row_id][col_id]

            # the "divider" column, leave blank
            if col_id == 1:
                ax.axis('off')
                continue

            # display target sprite for this column
            if col_id == 0:
                target_sprite = target_sprites[row_id]
                if sprite_type == "image":
                    ax.imshow(target_sprite, plt.get_cmap('binary'))
                elif sprite_type == "tabular":
                    sns.barplot(x_pos, target_sprite, palette='bright', ax=ax)
                elif sprite_type == "text":
                    ax.text(
                        0.5,
                        0.5,
                        target_sprite,
                        wrap=True,
                        va='center',
                        ha='center',
                        bbox=target_bbox,
                        fontsize=12)
                    ax.axis('off')

                # set the label/title
                target_label = y_targets[row_id]
                ax.title.set_text(target_label)
            else:
                # the rank of the closest sprite we want to display
                # (e.g) when rank == 0 we want to display the closest sprite
                rank = col_id - 2

                # find the label for this subplot
                test_id = indices[row_id][rank]
                test_label = y_test[test_id]

                # draw test sprite
                test_sprite = test_sprites[test_id]
                if sprite_type == "image":
                    # Change the colormap to Reds if this subplot is showing
                    # an incorrect nearest neighbor to highlight it.
                    if target_label == test_label:
                        ax.imshow(test_sprite, plt.get_cmap('binary'))
                    else:
                        ax.imshow(test_sprite, plt.get_cmap('Reds'))
                elif sprite_type == "tabular":
                    sns.barplot(x_pos, test_sprite, palette='bright', ax=ax)
                elif sprite_type == "text":
                    bbox = correct_label_bbox
                    if target_label != test_label:
                        bbox = incorrect_label_bbox
                    ax.text(
                        0.5,
                        0.5,
                        test_sprite,
                        wrap=True,
                        va='center',
                        ha='center',
                        bbox=bbox,
                        fontsize=12)
                    ax.axis('off')

                # set the label/title
                ax.title.set_text(test_label)

                # add distance as the x label
                distance = distances[row_id][rank]
                distance_message = "{:.4f}".format(distance)

                # only add the word "distance" on the closest sprites
                if rank == 0:
                    distance_message = "distance: " + distance_message

                ax.set_xlabel(distance_message)

                # Further highlight incorrect nearest neighbors
                if target_label != test_label:
                    ax.xaxis.label.set_color('red')
                    ax.title.set_color('red')
                    ax.spines['bottom'].set_color('red')
                    ax.spines['top'].set_color('red')
                    ax.spines['right'].set_color('red')
                    ax.spines['left'].set_color('red')

            # remove ticks and grid to declutter the figure
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

    # prevent overlaps
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])

    # close the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)

    return figure
