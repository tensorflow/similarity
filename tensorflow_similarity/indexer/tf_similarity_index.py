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

import argparse
from tensorflow_similarity.indexer.indexer import Indexer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--index_dir", required=True, help="index directory")
    ap.add_argument("-m", "--model", required=True, help="Tf.similarity model")
    ap.add_argument("-d", "--dataset", required=True, help="Dataset")
    args = vars(ap.parse_args())
    indexer = Indexer(args["dataset"], args["model"], args["index_dir"])

if __name__ == "__main__":
    main()