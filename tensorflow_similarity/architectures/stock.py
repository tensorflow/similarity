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

from tensorflow_similarity.architectures.model_registry import register_model
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def NoTopWrapper(fn):
    def f(input_shapes):
        shape = input_shapes['input_1']
        print("Shape; %s" % str(shape))

        load_weights = True
        if shape[0] != shape[1]:
            load_weights = False
        if shape[0] not in [96, 128, 160, 192, 224]:
            load_weights = False

        if load_weights:
            return fn(include_top=False, input_shape=shape, pooling="avg")
        else:
            return fn(
                include_top=False,
                weights=None,
                input_shape=shape,
                pooling="avg")
        return o

    return f


register_model(NoTopWrapper(DenseNet121), "DenseNet121")
register_model(NoTopWrapper(DenseNet169), "DenseNet169")
register_model(NoTopWrapper(DenseNet201), "DenseNet201")
register_model(NoTopWrapper(MobileNetV2), "MobileNetV2")
register_model(NoTopWrapper(MobileNet), "MobileNet")
