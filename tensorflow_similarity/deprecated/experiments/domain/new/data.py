#!python

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

import csv
import tensorflow as tf


def load_top_n(filename, N):
    output = []
    with tf.io.gfile.GFile(filename, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            output.append(row["IDN_Domain"])
            if len(output) == N:
                break

    return output


print(load_top_n("majestic_million.csv", 10))
