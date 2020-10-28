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

from tensorflow_similarity.callbacks.base import MoiraiCallback
from tensorflow_similarity.utils.config_utils import value_or_callable, deserialize_moirai_object, register_custom_object
import numpy as np
import os
import tensorflow as tf

import time


class RefreshStrategy(MoiraiCallback):
    def __init__(self, sampler=None, sentinel_file=None):
        super(RefreshStrategy, self).__init__()
        self.sampler = deserialize_moirai_object(sampler)
        self.sentinel_file = value_or_callable(sentinel_file)
        self.refresh_in_progress = False

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'sampler': self.sampler.get_config(),
            }
        }

    def asynchronous_refresh(self):
        self.refresh_in_progress = True
        self.sampler(callback=self.on_refresh_complete)

    def synchronous_refresh(self):
        self.sampler()
        self.on_refresh_complete([])

    def on_refresh_complete(self, _):
        self.update_sentinel()
        self.refresh_in_progress = False

    def update_sentinel(self):
        with tf.io.gfile.GFile(self.sentinel_file, "w") as f:
            f.write(str(time.time()))

    def process_epoch_end(self, batch, logs):
        """Overridden by subclasses to determine whether we should
        update the dataset."""
        raise NotImplementedError

    def on_epoch_end(self, epoch, logs={}):
        if not self.refresh_in_progress:
            self.process_epoch_end(epoch, logs)


class PeriodicRefreshStrategy(RefreshStrategy):
    def __init__(self, period=10, synchronous_refresh=False, count=0,
                 **kwargs):
        super(PeriodicRefreshStrategy, self).__init__(**kwargs)
        self.period = value_or_callable(period)
        self.count = count
        self.synchronous = value_or_callable(synchronous_refresh)

    def process_epoch_end(self, epoch, logs):
        self.count += 1
        if self.count % self.period == 0:
            if self.synchronous:
                self.synchronous_refresh()
            else:
                self.asynchronous_refresh()

    def get_config(self):
        config = super(PeriodicRefreshStrategy, self).get_config()
        config['config']['period'] = self.period
        config['config']['synchronous_refresh'] = self.synchronous
        config['config']['count'] = self.count
        return config


register_custom_object("PeriodicRefreshStrategy", PeriodicRefreshStrategy)


class LowLossRefreshStrategy(RefreshStrategy):
    def __init__(self,
                 threshold=.01,
                 metric='loss',
                 lower_is_better=True,
                 consecutive_epochs=2,
                 **kwargs):
        super(LowLossRefreshStrategy, self).__init__(**kwargs)

        self.metric = metric
        self.threshold = threshold
        self.consecutive_epochs = consecutive_epochs
        self.trailing_losses = []
        self.lower_is_better = lower_is_better

    def get_config(self):
        config = super(LowLossRefreshStrategy, self).get_config()
        config['config']['threshold'] = self.threshold
        config['config']['metric'] = self.metric
        config['config']['lower_is_better'] = self.lower_is_better
        config['config']['consecutive_epochs'] = self.consecutive_epochs
        return config

    def process_epoch_end(self, epoch, logs={}):
        if self.metric in logs:
            metric = logs[self.metric]
            is_below = metric < self.threshold
            if is_below == self.lower_is_better:
                self.losses.append(loss)
                if len(self.losses) == self.consecutive_batches:
                    self.losses = []
                    self.asynchronous_refresh()
            else:
                self.losses = []


register_custom_object("LowLossRefreshStrategy", LowLossRefreshStrategy)
