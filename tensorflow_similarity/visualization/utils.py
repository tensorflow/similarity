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

import numpy as np


def filter_data(x_data, y_labels, selected_classes, image_data=[]):
    """Filter out x_data and y_labels based on selected classes.

    Arguments:
        x_data (List[Embeddings]): The list of embeddings.
        y_labels (List[Integers]): The list of labels. Should be the same
            length as x_data and y_labels[i] is the label for x_data[i]
        selected_classes (Array[Integers]): The array-like parameter that
            specify which classes to return.
        image_data (List[Image Data]): The list of images (2D for black and
            white or 3D for RGB). Where y_labels[i] is the label for
            image_data[i]. When not None we will filter it as well.
            Defults to [].
    Returns:
        filtered_x_data (np.array), filtered_y_labels (np.array),
            filtered_image_data (np.array): A tuple of three arrays, each
                represents the filtered arrays (only the selected classes'
                data remains) for x_data, y_labels, and image_data respectively.
    """

    assert len(x_data) == len(y_labels)

    unique_classes = set(selected_classes)
    filtered_x_data = []
    filtered_y_labels = []
    filtered_image_data = []

    for i, label in enumerate(y_labels):
        if label in unique_classes:
            filtered_y_labels.append(label)
            filtered_x_data.append(x_data[i])

            if len(image_data) == len(y_labels):
                filtered_image_data.append(image_data[i])

    filtered_x_data = np.array(filtered_x_data)
    filtered_y_labels = np.array(filtered_y_labels)
    filtered_image_data = np.array(filtered_image_data)

    return filtered_x_data, filtered_y_labels, filtered_image_data


def make_ordinal(num):
    '''Convert an integer into its ordinal representation:

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    num = int(num)
    suffixes = ['th', 'st', 'nd', 'rd', 'th']
    suffix = suffixes[min(num % 10, 4)]

    # special case, when the number ends in 11,12,13, change the suffix to 'th'
    if 11 <= (num % 100) <= 13:
        suffix = 'th'

    return str(num) + suffix
