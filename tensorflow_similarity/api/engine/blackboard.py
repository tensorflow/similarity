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

import collections
import copy
import json
import numpy as np
import os
from threading import Lock, Thread


class BlackboardData(object):
    def __init__(self, modification_time=None, tuples=[]):
        self.modification_time = modification_time

        tmp = copy.copy(tuples)
        np.random.shuffle(tmp)
        self.tuples = collections.deque(tmp)

    def empty(self):
        return not self.tuples

    def sample(self):
        if self.empty():
            return None
        else:
            return self.tuples.pop()


class Blackboard(object):
    def __init__(self, filename, tower_names):
        self.filename = filename
        self.tower_names = tower_names
        self.modification_time = None
        self.data = BlackboardData(None, [])
        self.reader_thread = None

    def get(self):
        self.check_for_updates()
        return self.data

    def clear(self):
        self.data = BlackboardData(None, [])

    def check_for_updates(self):
        if not os.path.exists(self.filename):
            self.data = BlackboardData(None, [])
            return

        try:
            stat_result = os.stat(self.filename)
        except BaseException:
            # File may have been deleted in the interim. Either way, there is
            # nothing for us to read.
            return

        # Latest file has already been read.
        if (self.data is not None
                and stat_result.st_mtime == self.data.modification_time):
            return

        # Read in progress already.
        if (self.reader_thread and self.reader_thread.is_alive()):
            return

        if (self.reader_thread and not self.reader_thread.is_alive()):
            self.reader_thread = None

        if not self.reader_thread:
            self.reader_thread = Thread(target=self.get_blackboard_reader())
            self.reader_thread.start()

    def get_blackboard_reader(self):
        def read_blackboard():
            if os.path.exists(self.filename):
                stat_result = os.stat(self.filename)
                modification_time = stat_result.st_mtime
                with open(self.filename, "r") as f:
                    history = json.load(f)
                    idxs = [history['idx'][tower]
                            for tower in self.tower_names]
                    hard_tuple_indices = [i for i in zip(*idxs)]

                    blackboard = BlackboardData(
                        modification_time=modification_time,
                        tuples=hard_tuple_indices)

                    self.data = blackboard

        return read_blackboard
