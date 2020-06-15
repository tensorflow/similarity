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

from absl import app, flags
import time
import tensorflow as tf
import multiprocessing

flags.DEFINE_integer('epochs', 100, 'Number of epochs to train the data.')

flags.DEFINE_integer('batch_size', 128,
                     'Number of examples (or triplets) per batch.')

flags.DEFINE_integer('prewarm_epochs', 1,
                     'Number of epochs to train the data.')
flags.DEFINE_float(
    'step_size_multiplier', 0.1,
    'Multiplier used to determine how many times each example should '
    'be touched on average, for one epoch.'
)
flags.DEFINE_string('run_id', time.strftime("%Y%m%d_%H%M%S"),
                    'Identifier used for this run.')

flags.DEFINE_string(
    'validation_sets', "",
    "Comma separated list of validation set names to be read.")
flags.DEFINE_integer('num_gpus', 2,
                     'Number of GPUs to use for multi-gpu training.')
flags.DEFINE_string('output_dir', None,
                    'Base directory in which to write data.')
flags.DEFINE_string(
    'sentinel_file', None,
    'Temporary file used to communicate when the sampled data is updated.')
flags.DEFINE_string('input_file_pattern', None, 'Input files to read.')
flags.DEFINE_string('sample_file_pattern', None,
                    'Sampled input files to read.')

flags.DEFINE_string('validation_file_pattern', None,
                    'File glob containing validation data.')
flags.DEFINE_string('preprocessed_validation_file_pattern', None,
                    'Temporary file for preprocessed validation data.')
flags.DEFINE_boolean('callbacks_copy_weights', False,
                     'If true, copy weights to the inference model.')

flags.DEFINE_integer('generator_workers',
                     min(multiprocessing.cpu_count() - 2, 1), '')
flags.DEFINE_integer('neighborhood_workers', 1, '')
flags.DEFINE_integer('sample_workers', 1, '')
flags.DEFINE_integer(
    'refresh_period', 1,
    'Period (in epochs) between refreshes when using PeriodicRefreshStrategy')
flags.DEFINE_integer(
    'sample_size', 100,
    'Number of examples to sample from the dataset at a time.')
flags.DEFINE_boolean(
    'colorize', True,
    'If true, the job is allowed to use termcolor based color.')

flags.DEFINE_boolean(
    'using_remote_filesystem', False,
    'If true, assume that the job is using some non-native filesystem '
    '(e.g. gs://). Files that MUST be written to local disk (e.g. those '
    'written by libraries that do not understand said filesystems) will be '
    'written locally and then moved to the remote filesystem, via a copy+delete'
)
flags.DEFINE_string(
    'local_tmp_dir', './',
    'When --using_remote_filesystem, the local destination for '
    'files before they are copied to the remote system.')
flags.DEFINE_string(
    'tmp_dir', './',
    'When --using_remote_filesystem, the local destination for '
    'files before they are copied to the remote system.')
flags.DEFINE_integer(
    'hard_mining_queue_size', 150000,
    'Number of hard tuples to keep in the hard-tuple queue')
flags.DEFINE_float("recycle_hard_tuple_frequency", .5, "")
flags.DEFINE_string(
    "tabulate_table_format", "grid",
    "Table format to use with tabulate. See https://pypi.org/project/tabulate/ for more information."
)

flags.DEFINE_boolean("dump_generator_output", False,
                     "If True, dump the output of the input generator.")
flags.DEFINE_boolean("dump_losses", False, "If True, dump the loss data.")
