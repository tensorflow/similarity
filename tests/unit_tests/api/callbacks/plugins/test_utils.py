# Lint as: python3
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
"""This file contains unit test for moirai/api/callbacks/utils directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import mock
from mock import patch

# import to patch for unit-test
import tensorflow as tf

from matplotlib import pyplot as plt

from tensorflow_similarity.api.callbacks.plugins.utils import plot_to_tensor


class TestPluginUtils(unittest.TestCase):
    """Tests all utility methods for morari metrics callback plugins.
    """

    @patch.object(tf, 'expand_dims', return_value='image_tensor')
    @patch.object(tf.image, 'decode_png', return_value='decoded_image')
    def testPlotToTensor(self, mock_decode_png, mock_expand_dims):
        """Simple test that tests plot_to_tensor behaves as expected.

        Args:
            mock_decode_png (Mock): The MagicMock that patches
                tf.expand_dims and returns 'image_tensor'.
            mock_expand_dims (Mock): The magicMock that patches
                tf.image.decode_png and returns 'decoded_image'.
        """

        confusion_matrix = [[0, 2], [1, 0]]
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(confusion_matrix)
        image_tensor = plot_to_tensor(figure)

        mock_decode_png.assert_called_once()
        mock_expand_dims.assert_called_once_with('decoded_image', 0)
        self.assertEqual(image_tensor, 'image_tensor')


if __name__ == '__main__':
    unittest.main()
