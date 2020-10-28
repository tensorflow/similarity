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

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_similarity.utils.config_utils import register_custom_object
import collections
import six


class Pinned(object):
    def __init__(self, device=None, **kwargs):
        self.raw_device = device
        if device == 'any':
            self.device = '/cpu:0'
            for dev in K.get_session().list_devices():
                if dev.device_type == "GPU":
                    self.device = dev.name
        else:
            self.device = device

    def __call__(self, x):
        return self.device

    def get_config(self):
        return {
            'class_name': 'Pinned',
            'config': {
                'device': self.raw_device,
            },
        }


register_custom_object("Pinned", Pinned)


class CPU(Pinned):
    def __init__(self, **kwargs):
        self.device = "/cpu:0"

    def get_config(self):
        return {'class_name': 'CPU', 'config': {}}


register_custom_object("CPU", CPU)


class RoundRobin(object):
    def __init__(self, devices=None, **kwargs):
        self.raw_devices = devices
        if devices == 'all':
            devices = []
            for dev in K.get_session().list_devices():
                if dev.device_type == "GPU":
                    devices.append(dev.name)
            if len(devices) == 0:
                devices = ['/cpu:0']
        elif isinstance(devices, six.string_types):
            devices = devices.split(",")
        elif isinstance(devices, collections.Iterable):
            devices = [x for x in devices]
        else:
            print("%s is not a known device." % devices)
            raise ValueError

        self.devices = devices
        self.assigned = {}
        self.next_idx = 0

    def __call__(self, x):
        if x not in self.assigned:
            self.assigned[x] = self.devices[self.next_idx % len(self.devices)]
            self.next_idx += 1
        return self.assigned[x]

    def get_config(self):
        return {
            'class_name': 'RoundRobin',
            'config': {
                'devices': self.raw_devices,
            },
        }


register_custom_object("RoundRobin", RoundRobin)
