# Copyright 2021 The TensorFlow Authors
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
from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib import pyplot as plt

from tensorflow_similarity.types import Lookup, Tensor


def viz_neigbors_imgs(
    example: Tensor,
    example_class: int,
    neighbors: Sequence[Lookup],
    class_mapping: Mapping[int, str] | None = None,
    fig_size: tuple[int, int] = (24, 4),
    cmap: str = "viridis",
    show: bool = True,
):
    """Display images nearest neighboors

    Args:
        example: The data used as query input.

        example_class: The class of the data used as query

        neighbors: The list of neighbors returned by the lookup()

        class_mapping: Mapping from class numerical ids to a class name. If not
        set, the plot will display the class numerical id instead.
        Defaults to None.

        fig_size: Size of the figure. Defaults to (24, 4).

        cmap: Default color scheme for black and white images e.g mnist.
        Defaults to 'viridis'.

        show: If the plot is going to be shown or not. Defaults to True.
    """
    num_cols = len(neighbors) + 1
    _, axs = plt.subplots(1, num_cols, figsize=fig_size)

    # draw target
    axs[0].imshow(example, cmap=cmap)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    class_label = _get_class_label(example_class, class_mapping)
    axs[0].set_title(class_label)

    for ax, nbg in zip(axs[1:], neighbors):
        val = _get_class_label(nbg.label, class_mapping)
        legend = f"{val} - {nbg.distance:.5f}"
        if nbg.label == example_class:
            color = cmap
        else:
            color = "Reds"
        ax.imshow(nbg.data, cmap=color)
        ax.set_title(legend)
        ax.set_xticks([])
        ax.set_yticks([])
    if show:
        plt.show()
    else:
        return plt


def _get_class_label(example_class, class_mapping):
    if example_class is None:
        return "No Label"

    if class_mapping is None:
        return str(example_class)

    class_label = class_mapping.get(example_class)
    return class_label if class_label is not None else str(example_class)
