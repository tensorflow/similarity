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

import copy
import os

import six
import numpy as np
from tensorflow_similarity.api.engine.generator import Generator


class TaskBasedGenerator(Generator):
    def __init__(self, main_task, auxillary_tasks):
        self.main_task = main_task
        self.auxillary_tasks = auxillary_tasks
        self.seq_id = 0

    def __len__(self):
        return len(self.main_task.generator)

    def get_batch(self, seq_id):
        batch = self.main_task.get_main_batch(seq_id)
        for task in self.auxillary_tasks:
            task.update_batch(batch)
        return batch
