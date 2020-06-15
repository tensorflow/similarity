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

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_similarity.utils.config_utils import value_or_callable, register_custom_object
from tensorflow_similarity.callbacks.base import MoiraiCallback
import os


class WeightCopier(MoiraiCallback):
    def __init__(self, copy_weights=False):
        super(WeightCopier, self).__init__()
        self.copy_weights = value_or_callable(copy_weights)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'copy_weights': self.copy_weights,
            }
        }

    def on_epoch_end(self, batch, logs={}):
        if self.copy_weights:
            self.simhash.inference_model().set_weights(self.model.get_weights())


register_custom_object("WeightCopier", WeightCopier)
