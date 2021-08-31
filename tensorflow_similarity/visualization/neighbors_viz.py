from typing import Mapping, Optional, Sequence, Tuple

from matplotlib import pyplot as plt

from tensorflow_similarity.types import Tensor, Lookup


def viz_neigbors_imgs(example: Tensor,
                      example_class: int,
                      neighbors: Sequence[Lookup],
                      class_mapping: Optional[Mapping[int, str]] = None,
                      fig_size: Tuple[int, int] = (24, 4),
                      cmap: str = 'viridis'):
    """Display images nearest neighboors

    Args:
        example: The data used as query input.

        example_class: The class of the data used as query

        neighbors: The list of neighbors returned by the lookup()

        class_mapping: Dictionary that map the class numerical ids to a class
        name. If not set, will display the class numerical id.
        Defaults to None.

        fig_size: Size of the figure. Defaults to (24, 4).

        cmap: Default color scheme for black and white images e.g mnist.
        Defaults to 'viridis'.
    """
    num_cols = len(neighbors) + 1
    plt.subplots(1, num_cols, figsize=fig_size)
    plt_idx = 1

    # draw target
    plt.subplot(1, num_cols, plt_idx)
    plt.imshow(example, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    val = class_mapping[example_class] if class_mapping else str(example_class)
    plt.title(val)
    plt_idx += 1

    for nbg in neighbors:
        plt.subplot(1, num_cols, plt_idx)
        if class_mapping and nbg.label is not None:
            val = class_mapping[nbg.label]
        elif nbg.label is not None:
            val = str(nbg.label)
        else:
            val = 'No Label'
        legend = f"{val} - {nbg.distance:.5f}"
        if nbg.label == example_class:
            color = cmap
        else:
            color = 'Reds'
        plt.imshow(nbg.data, cmap=color)
        plt.title(legend)
        plt.xticks([])
        plt.yticks([])

        plt_idx += 1
    plt.show()
