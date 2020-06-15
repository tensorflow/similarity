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


class MoiraiInitializer(object):
    def __init__(self, train_only=True):
        self.train_only = train_only

    def __call__(self, is_training=True, **kwargs):
        print("Maybe: %s" % self.__class__.__name__)
        if is_training or not self.train_only:
            self.run_initializer(**kwargs)

    def run_initializer(self):
        raise NotImplementedError

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'train_only': self.train_only
            }
        }
