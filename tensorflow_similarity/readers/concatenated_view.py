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

import bisect
import numpy as np


class ConcatenatedView(object):
    """Wrapper around a list of array-likes (e.g. np.array or HDF5Matrix) which
    provides a view of the data, without copying the entirety of the dataset in
    memory.
    """

    def __init__(self, shards):
        self.shards = shards
        self.length = None
        self.lengths = []
        self.start_offsets = []
        self.end_offsets = []
        self.size = None

        self._compute_lengths()
        self._compute_size()

    def _compute_lengths(self):
        if self.length:
            return
        total_length = 0
        for shard in self.shards:
            l = len(shard)
            self.lengths.append(l)
            self.start_offsets.append(total_length)
            total_length += l
            self.end_offsets.append(total_length)
        self.length = total_length

    def __len__(self):
        return self.length

    def _compute_size(self):
        if self.size:
            return
        s = 0
        for shard in self.shards:
            s += shard.size
        self.size = s

    @property
    def shape(self):
        return (len(self), +self.shards[0]._base_shape)

    @property
    def dtype(self):
        return self.shards[0].dtype

    @property
    def ndim(self):
        return self.shards[0].ndim

    def shard_for_id(self, id, direction="left"):
        if direction == "left":
            shard = bisect.bisect_left(self.end_offsets, id)
        else:
            assert direction == "right"
            shard = bisect.bisect_right(self.end_offsets, id)

        if shard == len(self.end_offsets):
            return IndexError
        return shard

    def get_from_shard(self, shard, global_start, global_stop):
        start_offset = self.start_offsets[shard]
        start = global_start - start_offset
        stop = global_stop - start_offset
        return self.shards[shard][start:stop]

    def get_shard(self, shard):
        return self.shards[shard][:]

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = len(self) - 1
            if stop > len(self):
                raise IndexError("Index %s is out of range for [0,%d)" %
                                 (str(key), len(self)))

            start_shard = self.shard_for_id(start)
            stop_shard = self.shard_for_id(stop)

            return_data = []
            if start_shard == stop_shard:
                return_data.append(
                    self.get_from_shard(start_shard, start, stop))
            else:
                return_data.append(
                    self.get_from_shard(start_shard, start,
                                        self.end_offsets[start_shard]))

            # Shards that aren't start or stop return all data
            for shard in range(start_shard + 1, stop_shard):
                return_data.append(self.get_shard(shard))

            if start_shard != stop_shard:
                return_data.append(
                    self.get_from_shard(stop_shard,
                                        self.start_offsets[stop_shard], stop))

            return np.concatenate(return_data, axis=0)

            # Break slice into one slice per shard
            # Read each subslice
            # Join
        elif isinstance(key, (int, np.integer)):
            shard = self.shard_for_id(key, direction="right")
            return self.shards[shard][key - self.start_offsets[shard]]
        else:
            output = np.zeros([len(key)] + list(self.shards[0].shape)[1:])
            for output_idx, data_idx in enumerate(key):
                output[output_idx] = self[data_idx]
            return output
