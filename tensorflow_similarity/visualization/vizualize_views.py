from matplotlib import pyplot as plt
from tensorflow import Tensor
import tensorflow as tf
from typing import Tuple


def visualize_views(views: Tensor,
                    labels: Tensor = None,
                    predictions: Tensor = None,
                    num_imgs: int = None,
                    views_per_col: int = 4,
                    fig_size: Tuple[int, int] = (24, 4),
                    max_pixel_value: float = 1.0):
    """Display side by side different image views with labels, and predictions

    Args:
        views: Aray of views
        predictions: model output.
        labels: image labels
        num_imgs: number of images to use.
        views_per_col: Int, number of images in one row. Defaults to 3.
        max_pixel_value: Max expected value for a pixel. Used to scale the image
          between [0,1].

    Returns:
        None.
    """
    num_views = len(views)
    num_imgs = num_imgs if num_imgs else len(views[0])
    num_col = views_per_col
    num_row = num_imgs // num_col
    num_row = num_row + 1 if num_imgs % num_col else num_row

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=fig_size)
    for i in range(num_imgs):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        pair = [views[j][i] / max_pixel_value for j in range(num_views)]
        ax.imshow(tf.concat(pair, axis=1))
        ax.set_axis_off()

        label = labels[i] if labels else i

        if predictions:
            ax.set_title("Label: {} | Pred: {:.5f}".format(label,
                                                           predictions[i][0]))
        elif labels:
            ax.set_title("Label: {}".format(label))


        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
