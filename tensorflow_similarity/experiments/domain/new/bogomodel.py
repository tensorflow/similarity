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

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow_similarity.experiments.domain.new.dependencies import FakeHash


i = Input(name='example', shape=(1,), dtype=tf.string)
o = FakeHash()(i)

m = Model(i, o)
m.compile(loss="mse", optimizer="adam")

x = m.predict(np.array([["google.com"], ["facebook.com"]]))
print(x)
print(np.shape(x))

m.save("bogo.model.checkpoint")
