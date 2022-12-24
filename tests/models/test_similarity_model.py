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
import tensorflow as tf

from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.models import SimilarityModel


class SimilarityModelTest(tf.test.TestCase):
    # TODO(ovallis): Add tests for graph mode.
    def test_save_and_reload(self):
        out_dir = self.get_temp_dir()

        inputs = tf.keras.layers.Input(shape=(3,))
        outputs = tf.keras.layers.Dense(2)(inputs)
        model = SimilarityModel(inputs, outputs)
        model.compile(optimizer="adam", loss=TripletLoss())

        # index data
        x = tf.constant([[1, 1, 3], [3, 1, 2]], dtype="float32")
        y = tf.constant([1, 2])
        model.index(x, y)

        # save
        model.save(out_dir)

        # reload
        loaded_model = tf.keras.models.load_model(out_dir)
        loaded_model.load_index(out_dir)
        self.assertEqual(loaded_model._index.size(), len(y))

    def test_save_no_compile(self):
        out_dir = self.get_temp_dir()

        inputs = tf.keras.layers.Input(shape=(3,))
        outputs = tf.keras.layers.Dense(2)(inputs)
        model = SimilarityModel(inputs, outputs)

        model.save(out_dir)
        model2 = tf.keras.models.load_model(out_dir)
        self.assertIsInstance(model2, type(model))

    def test_index_single(self):
        """Unit Test for issues #161 & #162"""
        inputs = tf.keras.layers.Input(shape=(3,))
        outputs = tf.keras.layers.Dense(2)(inputs)
        model = SimilarityModel(inputs, outputs)
        model.compile(optimizer="adam", loss=TripletLoss())

        # index data
        x = tf.constant([1, 1, 3], dtype="float32")
        y = tf.constant([1])

        # run individual sample & index
        model.index_single(x, y, data=x)
        self.assertEqual(model._index.size(), 1)


if __name__ == "__main__":
    tf.test.main()
