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

from tensorflow_similarity.losses import SimSiamLoss
from tensorflow_similarity.models import ContrastiveModel, create_contrastive_model


class ContrastiveModelTest(tf.test.TestCase):
    def test_save_and_reload(self):
        """Test save and load of ContrastiveModel.
        Testing it also in a MirroredStrategy on GPU if available, to check fix for
        issue #287.
        """
        out_dir = self.get_temp_dir()
        # with tf.distribute.MirroredStrategy().scope():
        backbone_input = tf.keras.layers.Input(shape=(3,))
        backbone_output = tf.keras.layers.Dense(4)(backbone_input)
        backbone = tf.keras.Model(
            inputs=backbone_input,
            outputs=backbone_output,
        )

        projector_input = tf.keras.layers.Input(shape=(4,))
        projector_output = tf.keras.layers.Dense(4)(projector_input)
        projector = tf.keras.Model(
            inputs=projector_input,
            outputs=projector_output,
        )

        predictor_input = tf.keras.layers.Input(shape=(4,))
        predictor_output = tf.keras.layers.Dense(4)(predictor_input)
        predictor = tf.keras.Model(
            inputs=predictor_input,
            outputs=predictor_output,
        )

        model = create_contrastive_model(
            backbone=backbone,
            projector=projector,
            predictor=predictor,
            algorithm="simsiam",
        )
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.5)

        model.compile(optimizer=opt, loss=SimSiamLoss())

        # test data
        x = tf.constant([[1, 1, 3], [3, 1, 2]], dtype="int64")

        # create dataset with two views
        ds = tf.data.Dataset.from_tensors(x)
        ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)

        model.fit(ds)

        # save
        model.save(out_dir)

        # reload and test loaded model
        # with tf.distribute.MirroredStrategy().scope():
        loaded_model = tf.keras.models.load_model(
            out_dir,
            custom_objects={"ContrastiveModel": ContrastiveModel},
        )

        pred = loaded_model.predict(x)

        self.assertEqual(loaded_model.algorithm, "simsiam")
        self.assertEqual(loaded_model.optimizer.lr, 0.5)
        self.assertAllEqual(loaded_model.backbone.input_shape, (None, 3))
        self.assertAllEqual(loaded_model.backbone.output_shape, (None, 4))
        self.assertAllEqual(loaded_model.predictor.input_shape, (None, 4))
        self.assertAllEqual(loaded_model.predictor.output_shape, (None, 4))
        self.assertAllEqual(loaded_model.projector.input_shape, (None, 4))
        self.assertAllEqual(loaded_model.projector.output_shape, (None, 4))
        self.assertAllEqual(pred.shape, (2, 4))
        self.assertAllEqual(model.predict(x), pred)


if __name__ == "__main__":
    tf.test.main()
