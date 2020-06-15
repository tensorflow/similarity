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

from tensorflow.keras.callbacks import Callback


class MoiraiCallback(Callback):
    def __init__(self):
        super(MoiraiCallback, self).__init__()

        self.simhash = None

    # TODO(b/142676963): After the training loop rewrite with
    # tf.graidentTape is completed we should no longer need to set simhash.
    def set_simhash(self, simhash):
        self.simhash = simhash

    def set_model(self, model):
        self.model = model
