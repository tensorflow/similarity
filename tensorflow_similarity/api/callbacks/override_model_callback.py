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


class OverrideModelCallback(Callback):
    """Sets the active model for each of the callbacks to the specified model.

    Most callbacks that make use of the Model will want the inference model,
    as it operates on a single callback at a time, and is the version that
    would typically be used for inference, saving, etc.
    """

    def __init__(self, simhash, other_callbacks):
        """Constructs the model override callback.

        Args:
            simhash (SimHash): The SimHash model to set for each callbacks.
            other_callbacks (list[Callback]): The callback(s) to update.
        """

        # TODO(b/142676963): After the training loop rewrite with
        # tf.graidentTape is completed OverrideModelCallback should no
        # longer be necessary.

        self.callbacks = other_callbacks
        self.simhash = simhash

    def override_model(self):
        for callback in self.callbacks:
            if hasattr(callback, 'set_simhash'):
                callback.set_simhash(self.simhash)

    def on_train_begin(self, logs={}):
        self.override_model()
