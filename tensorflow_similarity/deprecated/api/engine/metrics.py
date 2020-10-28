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

from tensorflow_similarity.callbacks.validation_set_metrics import _ComputeValidationSetMetricsTask
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.ops import array_ops, init_ops, math_ops
from tensorflow.keras import backend as K


class EvalMetric(object):
    def __init__(self,
                 x_test=[],
                 y_test=[],
                 e_test=[],
                 x_targets=[],
                 y_targets=[],
                 e_targets=[]):
        self.x_test = x_test
        self.y_test = y_test
        self.e_test = e_test

        self.x_targets = x_targets
        self.y_targets = y_targets
        self.e_targets = e_targets

    def compute(self):
        """Returns a dictionary of metric name to float value."""
        raise NotImplementedError


class MoiraiMetrics(EvalMetric):
    def compute(self):
        return _ComputeValidationSetMetricsTask(
            "validation_set",
            self.e_test,
            self.y_test,
            self.e_targets,
            self.y_targets).process()[0]


class Pseudometric(Metric):
    """A Pseudometric is a metric that isn't a metric - we use this to pass data from training through to the callbacks """

    def __init__(self, tensor, **kwargs):
        super(Pseudometric, self).__init__(name="pseudometric", **kwargs)
        self.tensor_to_return = tensor
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        return array_ops.identity(self.tensor_to_return)

    def result(self):
        return array_ops.identity(self.tensor_to_return)


def fraction(y_true, y_pred):
    return K.mean(math_ops.abs(y_pred - y_true), axis=-1)
