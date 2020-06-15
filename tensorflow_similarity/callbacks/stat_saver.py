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

import json
from tensorflow_similarity.utils.config_utils import value_or_callable, register_custom_object
from tensorflow_similarity.callbacks.base import MoiraiCallback
import numpy as np
import os
import six
import traceback
import tensorflow as tf


class StatSaver(MoiraiCallback):
    def __init__(self, output_dir=None):
        super(StatSaver, self).__init__()
        self.output_dir = value_or_callable(output_dir)
        self.filename = os.path.join(self.output_dir, "model.validation_stats")
        self.history = []

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'output_dir': self.output_dir,
            }
        }

    def on_epoch_end(self, batch, logs={}):
        cleaned_logs = {}
        for k, v in six.iteritems(logs):
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, np.float16):
                v = float(v)
            if isinstance(v, np.float32):
                v = float(v)
            if isinstance(v, np.float64):
                v = float(v)

            cleaned_logs[k] = v
        self.history.append(cleaned_logs)

        try:

            with tf.io.gfile.GFile(self.filename, "w") as f:
                f.write(json.dumps(self.history))
        except BaseException:
            print("Failed to write stats to %s." % self.filename)
            print(self.history)
            traceback.print_exc()


register_custom_object("StatSaver", StatSaver)
