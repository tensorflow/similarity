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
import copy
import tensorflow as tf


class MultiGPUSafeCheckpoint(MoiraiCallback):
    def __init__(self,
                 output_dir=None,
                 tmp_dir=None,
                 filename="keras_checkpoint.{epoch:03d}-{loss:.5f}.hdf5",
                 **kwargs):
        super(MultiGPUSafeCheckpoint, self).__init__()
        self.output_dir = value_or_callable(output_dir)
        self.tmp_dir = value_or_callable(tmp_dir)
        self.filename = value_or_callable(filename)
        self.checkpoint_config = {}
        self.simhash = None

        if tmp_dir is not None:
            self.write_path = os.path.join(self.tmp_dir, self.filename)
            self.move_path = os.path.join(self.output_dir, self.filename)
        else:
            self.write_path = os.path.join(self.output_dir, self.filename)
            self.move_path = None

        for k, v in kwargs.items():
            self.checkpoint_config[k] = value_or_callable(v)
        self.saver = ModelCheckpoint(self.write_path, **self.checkpoint_config)

    def set_simhash(self, simhash):
        self.simhash = simhash

    def get_config(self):
        config = {
            'class_name': self.__class__.__name__,
            'config': {
                'output_dir': self.output_dir,
                'tmp_dir': self.tmp_dir,
                'filename': self.filename,
            }
        }

        for k, v in self.checkpoint_config.items():
            config['config'][k] = v

        return config

    def on_epoch_end(self, epoch, logs={}):
        write_file = self.write_path.format(epoch=epoch + 1, **logs)
        move_file = self.move_path.format(
            epoch=epoch + 1, **logs) if self.move_path else None

        self.saver.set_model(self.simhash.inference_task.task_model)
        self.saver.on_epoch_end(epoch, logs)
        if move_file is not None:
            tf.io.gfile.Copy(write_file, move_file, overwrite=True)
            tf.io.gfile.Remove(write_file)


register_custom_object("MultiGPUSafeCheckpoint", MultiGPUSafeCheckpoint)
