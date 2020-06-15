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

import tensorflow as tf
from tensorflow_similarity.callbacks.base import MoiraiCallback
from tensorflow_similarity.utils.config_utils import register_custom_object


class CallbackWrapper(MoiraiCallback):
    def __init__(self, cls=None, **kwargs):
        self.callback = cls(**kwargs)
        self.config = kwargs

    def set_model(self, model):
        self.callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        self.callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs)

    def get_config(self):
        return {'class_name': self.__class__.__name__, 'config': self.config}


class ReduceLROnPlateauWrapper(CallbackWrapper):
    def __init__(self, **kwargs):
        super(ReduceLROnPlateauWrapper, self).__init__(
            cls=tf.keras.callbacks.ReduceLROnPlateau, **kwargs)


register_custom_object("ReduceLROnPlateauWrapper", ReduceLROnPlateauWrapper)


class TensorboardWrapper(CallbackWrapper):
    def __init__(self, **kwargs):
        super(TensorboardWrapper, self).__init__(
            cls=tf.keras.callbacks.TensorBoard, **kwargs)


register_custom_object("TensorboardWrapper", TensorboardWrapper)
