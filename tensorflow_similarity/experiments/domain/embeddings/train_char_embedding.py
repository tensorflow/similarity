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
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

with open("/tmp/vis_embed", "rt") as f:
    char_dict = json.load(f)

chars = char_dict.keys()
x = np.array([char_dict[k] for k in chars])

for bottleneck_size in [16, 32, 64]:
    i = Input(shape=(64 * 64, ), name='img')
    e = Dense(64 * 16)(i)
    e = Dense(64 * 4)(e)

    bottleneck = Dense(bottleneck_size)(e)

    d = Dense(64 * 4)(bottleneck)
    d = Dense(64 * 16)(d)
    recon = Dense(64 * 64)(d)

    training = Model(inputs=i, outputs=recon)
    training.compile(optimizer="adam", loss="mse")

    inference = Model(inputs=i, outputs=bottleneck)
    inference.compile(optimizer="adam", loss="mse")

    training.fit(x, x, epochs=20)
    embedding = inference.predict(x)

    output_dict = {}
    for char, embedding in zip(chars, embedding):
        output_dict[char] = embedding.tolist()
    with open("/tmp/vis_embed_trained_%d" % bottleneck_size, "wt") as f:
        json.dump(output_dict, f)
