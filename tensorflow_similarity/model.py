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

from absl import app, flags
import json
from tensorflow_similarity.utils.config_utils import deserialize_moirai_object, register_custom_object
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers import serialize
from tensorflow.python import debug as tf_debug
from tensorflow.keras.optimizers import Adam
import warnings

FLAGS = flags.FLAGS


# TODO - move to a common location, DRY
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MoiraiModel(tensorflow.keras.models.Model):
    """Specialization of a keras Model which contains multiple variations of the model
    and configuration related to Triplet Loss."""

    def __init__(self, **kwargs):
        super(MoiraiModel, self).__init__(**kwargs)
        self.inference_model = None
        self.prewarm_model = None
        self.moirai_config = {}

    def set_moirai_config(self, config):
        self.moirai_config = config

    def set_inference_model(self, model):
        print("Set inference model on %s to %s" % (self, model))
        self.inference_model = model

    def get_inference_model(self):
        return self.inference_model or self

    def set_prewarm_model(self, model):
        print("Set prewarm model on %s to %s" % (self, model))
        self.prewarm_model = model

    def get_prewarm_model(self):
        return self.prewarm_model

    def get_config(self):
        base_config = super(MoiraiModel, self).get_config()

        # The config can be a _DictWrapper from the internals of Tensorflow. We
        # need it to be a simple dict here.
        cfg = {}
        for key in self.moirai_config:
            cfg[key] = self.moirai_config[key]
        base_config['moirai_config'] = cfg
        if self.inference_model and self.inference_model != self:
            base_config['inference_model'] = self.inference_model.get_config()
        # Ignoring the prewarm model for now because under the assumption that
        # if you're loading a model, you likely already warmed it. We could
        # make assumptions about the losses of the pre-warm model, or serialize
        # that information, but thus far it's not worth the effort.
        return base_config

    @classmethod
    def from_config(cls, config):
        model = super(MoiraiModel, cls).from_config(config)
        model.set_moirai_config(config['moirai_config'] if 'moirai_config' in
                                config else {})
        if 'inference_model' in config:
            inference_model = tensorflow.keras.models.Model.from_config(
                config['inference_model'])
            inference_model.compile(optimizer=Adam(lr=.0001), loss="mse")
            model.set_inference_model(inference_model)

        return model


register_custom_object("MoiraiModel", MoiraiModel)


class Moirai(object):
    def __init__(self,
                 config=None,
                 training_model=None,
                 multigpu_training_model=None,
                 inference_model=None,
                 tower_model=None,
                 generator=None,
                 generator_workers=0,
                 preprocessing=None,
                 callbacks=[],
                 initializers=[],
                 output_dir=None):
        self.config = config
        self.training_model = training_model
        self.multigpu_training_model = multigpu_training_model
        self.inference_model = inference_model
        self.tower_model = tower_model
        self.generator = generator
        self.generator_workers = generator_workers
        self.callbacks = callbacks
        self.initializers = initializers
        self.preprocessing = preprocessing
        self.output_dir = output_dir

    def get_output_dir(self):
        return self.output_dir

    def get_multigpu_training_model(self):
        return self.multigpu_training_model

    def get_training_model(self):
        return self.training_model

    def get_prewarm_model(self):
        if not self.training_model:
            return None
        return self.training_model.get_prewarm_model()

    def get_tower_model(self):
        return self.tower_model

    def get_inference_model(self):
        return self.inference_model

    def get_config(self):
        out = self.config.copy()
        out['model'] = serialize(self.tower_model)
        return out

    def get_generator(self):
        return self.generator

    def get_preprocessor(self):
        if not self.preprocessing:
            return lambda x: x
        else:
            return self.preprocessing

    def write_config(self, filename="model.config"):
        with tf.io.gfile.GFile("%s/%s" % (self.output_dir, filename), "w") as f:
            json.dump(self.get_config(), f, cls=NumpyEncoder)

    def train(self, epochs=100, prewarm_epochs=10):
        with warnings.catch_warnings():
            # Validation can be slow, but there's no need to spam about it.
            warnings.simplefilter("ignore", UserWarning)
            for initializer in self.initializers:
                initializer()

            start_epoch = 0
            training_model = self.get_training_model()
            prewarm_model = self.get_prewarm_model()
            if prewarm_model:
                print("Pre-warming the model for %d epochs." % prewarm_epochs)
                prewarm_model.fit_generator(
                    self.generator,
                    steps_per_epoch=self.generator.steps_per_epoch(),
                    epochs=prewarm_epochs,
                    use_multiprocessing=self.generator_workers > 1,
                    max_queue_size=(self.generator_workers + 1) * 16,
                    workers=self.generator_workers
                    if self.generator_workers > 1 else 0)

            return training_model.fit_generator(
                self.generator,
                steps_per_epoch=self.generator.steps_per_epoch(),
                epochs=epochs,
                callbacks=self.callbacks,
                use_multiprocessing=self.generator_workers > 1,
                max_queue_size=(self.generator_workers + 1) * 16,
                workers=self.generator_workers
                if self.generator_workers > 1 else 0,
                initial_epoch=prewarm_epochs)

    def get_callbacks(self):
        return self.callbacks

    @classmethod
    def create_inference_preprocessor(cls, config):
        preprocessing = None
        print("Preprocessing")
        if 'preprocessing' in config:
            preprocessing = deserialize_moirai_object(config['preprocessing'])
        return preprocessing

    @classmethod
    def load(cls, config_filename, model_filename, inference_only=False):
        model = load_model(model_filename)  # pcompile=False)

        with tf.io.gfile.GFile(config_filename, "r") as f:
            config = json.load(f)

        if inference_only:
            pp = cls.create_inference_preprocessor(config)
            return Moirai(
                config=config,
                multigpu_training_model=model,
                training_model=model,
                inference_model=model,
                tower_model=model,
                generator=None,
                callbacks=[],
                preprocessing=pp,
                output_dir=FLAGS.output_dir)

        def tower(_):
            return model

        return cls.create(config, tower, inference_only=inference_only)

    @classmethod
    def create(cls, config, tower_model, inference_only=False):
        output_dir = config['output_dir']

        initializers = []
        if 'initializers' in config:
            for initializer in config['initializers']:
                initializer = deserialize_moirai_object(initializer)
                initializers.append(initializer)
                # Initializers are run at the beginning, as they may produce data used
                # by e.g. the generator or callbacks.
                initializer(is_training=not inference_only)

        preprocessing = None
        if 'preprocessing' in config:
            preprocessing = deserialize_moirai_object(config['preprocessing'])

        callbacks = []
        generator = None

        if not inference_only:
            # PeriodicRefresh callback may produce the initial dataset, so we
            # instantiate the callbacks first. Callbacks which need access to
            # the model can implement set_inference_model() or
            # set_training_model()
            if 'callbacks' in config:
                for callback in config['callbacks']:
                    callbacks.append(deserialize_moirai_object(callback))

            generator = deserialize_moirai_object(config['generator'])
            tower_model = tower_model(generator.get_input_shape())
        else:
            tower_model = tower_model(config['input_shape'])

        strategy = deserialize_moirai_object(config['strategy'])

        if inference_only:
            inference_model = strategy.build(tower_model, training=False)
            training_model = None
        else:
            training_model, inference_model = strategy.build(
                tower_model, training=True)
            for callback in callbacks:
                if isinstance(callback, MoiraiCallback):
                    callback.set_inference_model(inference_model)
                    callback.set_training_model(training_model)

        return Moirai(
            config=config,
            multigpu_training_model=training_model,
            training_model=training_model,
            inference_model=inference_model,
            tower_model=tower_model,
            generator=generator,
            callbacks=callbacks,
            preprocessing=preprocessing,
            output_dir=output_dir)
