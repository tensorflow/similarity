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

import json
import numpy as np
import jsonlines
from collections.abc import Iterable

def load_packaged_dataset(dataset, dataset_labels, dict_key):
    """ load a dataset from json lines files
    
        Args:
            dataset (string): The path to the json lines file containing the dataset that should be loaded
            dataset_labels (string): The path to the json lines file containing the labels that should be loaded
            dict_key (string): The dictionary key of the dict that must be passed to the model

        Returns:
            packaged_x (dict): A dict containting the datasets encoded as np arrays
            data_y (np.ndarray): The labels for the dataset encoded as np arrays

    """
    data_x = np.asarray(read_json_lines(dataset))
    data_y = np.asarray(read_json_lines(dataset_labels)).flatten()
    packaged_x = {dict_key: data_x}

    return packaged_x, data_y


def read_json_lines(file):
    """ read a json lines file
        
        Args:
            file (string): The path to the json lines file to be read

        Returns:
            data (list): The decoded json lines file
    """
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
            
    return data


def write_json_lines(file, data):
    """ write data to a json lines file

        Args:
            file (string): The path to the json lines file that should be written to
            data (JSON serializable object): The data that should be written to the file
    """
    with jsonlines.open(file, mode='w') as writer:
        if isinstance(data, Iterable):
            writer.write_all(data)
        else:
            writer.write(data)

def write_json_lines_dict(file, data):
    """ write a dict to a json lines file

        Args:
            file (string): The path to the json lines file that should be written to
            data (dict): The dict that should be written to the file
    """
    with open(file, 'w') as writer:
        json.dump(data, writer)