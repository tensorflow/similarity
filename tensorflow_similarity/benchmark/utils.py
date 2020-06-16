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

"""This file contains utility methods for benchmark."""

from __future__ import absolute_import, division, print_function

import json
import itertools
import copy
import os
from datetime import datetime


def read_config_file(filepath):
    """Read in a config file and return a list of dictionaries.

    Args:
        filepath (String): Path to the config file to be read.

    Returns:
        configs (List[Config Dict]): List of config dictionaries specified by
            the input config file. This method will be update to use Apache Beam.
    """

    current_path = os.getcwd()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "benchmark_{}".format(current_time)
    output_path = os.path.join(current_path, log_dir)

    os.mkdir(output_path)

    with open(filepath) as fp:
        raw_config = json.load(fp)

    raw_config["output_path"] = output_path

    configs = []

    # make combinations of experiments based on those parameters
    architectures = raw_config.pop("architectures", ["VGG16"])
    embedding_sizes = raw_config.pop("embedding_sizes", [32])
    similarity_losses = raw_config.pop("similarity_losses", ["triplet_loss"])
    percentage_training_classes = raw_config.pop(
        "percentage_training_classes", [1.0])
    datasets = raw_config.pop("datasets", ["omniglot"])

    hard_minings = raw_config.pop("hard_minings", [False])

    products = itertools.product(
        architectures,
        embedding_sizes,
        similarity_losses,
        percentage_training_classes,
        datasets,
        hard_minings)

    if True in hard_minings:
        print("Not benchmarking with hardmining becaue it does not work yet.")
        hard_minings.remove(True)

    for product in products:

        (architecture, embedding_size, similarity_loss,
         percentage_training_class, dataset, hard_mining) = product

        config = copy.deepcopy(raw_config)
        config["tower_model_architecture"] = architecture
        config["similarity_loss"] = similarity_loss
        config["embedding_size"] = embedding_size
        config["percentage_training_class"] = percentage_training_class
        config["embedding_size"] = embedding_size
        config["dataset"] = dataset
        config["hard_mining"] = hard_mining

        configs.append(config)

    return configs
