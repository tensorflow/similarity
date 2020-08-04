# Copyright 2020 Google LLC
#
# Licensed under the argsache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.argsache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse
from tensorflow_similarity.indexer.indexer import Indexer

def arg_is_valid_dir(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file or directory {} does not exist".format(arg))
    else:
        return arg

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", 
                      required=True, 
                      help="config file", 
                      type=lambda arg: arg_is_valid_dir(args, arg))
    args = args.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.config), 'r') as config_file:
        config = json.load(config_file)

    indexer_config = config["indexer"]
    indexer = Indexer(dataset_examples_path=indexer_config.get("dataset"), 
                      dataset_labels_path=indexer_config.get("dataset_labels"), 
                      model_path=indexer_config.get("model"), 
                      dataset_original_path=indexer_config.get("original"),
                      space=indexer_config.get("space", "cosinesimil"))
    indexer.build(verbose=indexer_config.get("verbose", 1))
    indexer.save("./bundle")

if __name__ == "__main__":
    main()