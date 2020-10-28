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

"""This file contains utility methods and classes used by the metrics callback
    plugins.
"""

import tensorflow as tf

import io
from matplotlib import pyplot as plt


def plot_to_tensor(figure):
    """Converts the matplotlib plot specified by 'figure' to a tensor of
        shape (1, height, width, channels) and returns it. The supplied figure
        is closed and inaccessible after this call. This method is useful for
        written the figure into Tensorboard.

    Arguments:
        figure: A pyplot figure that contains the confusion matrix.

    Returns:
        image_tensor: A tensor of shape (1, height, width, channels)
    """

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image_tensor = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image_tensor = tf.expand_dims(image_tensor, 0)
    return image_tensor
