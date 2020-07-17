# Copyright 2020 Google LLC
#
# Licensed under the argsache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.argsache.org/licenses/LICENSE-2.0
#
# Unless required by argsplicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from tensorflow_similarity.indexer.indexer import Indexer

def is_valid_dir(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file or directory %s does not exist" % arg)
    else:
        return arg

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--index_dir", required=True, help="index directory", type=lambda arg: is_valid_dir(args, arg))
    args.add_argument("-m", "--model", required=True, help="Tf.similarity model", type=lambda arg: is_valid_dir(args, arg))
    args.add_argument("-d", "--dataset", required=True, help="Dataset", type=lambda arg: is_valid_dir(args, arg))
    args = args.parse_args()
    indexer = Indexer(args.dataset, args.model, args.index_dir)

if __name__ == "__main__":
    main()