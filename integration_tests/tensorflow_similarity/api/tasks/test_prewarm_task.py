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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.python import debug as tf_debug

from tensorflow_similarity.api.engine.generator import Batch
from tensorflow_similarity.api.engine.task import MetaTask
from tensorflow_similarity.api.generators.prewarm import PrewarmGeneratorWrapper
from tensorflow_similarity.api.generators.task_based_generator import TaskBasedGenerator
from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask
from tensorflow_similarity.api.tasks.prewarm_task import PrewarmTask, create_prewarming_task
from tensorflow_similarity.api.tasks.quadruplet_loss_task import QuadrupletLossTask
from tensorflow_similarity.api.tasks.utils_for_test import *

# TODO - proper integration tests for prewarming.
