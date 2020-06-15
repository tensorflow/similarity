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
import json
import tensorflow_similarity
from tensorflow_similarity import trainer
from tensorflow_similarity import common_flags
from tensorflow_similarity.filters.text_distance import EditDistanceFilter
from tensorflow_similarity.utils.config_utils import load_custom_configuration, json_dict_to_moirai_obj, load_json_configuration_file, get_module_version
import multiprocessing
import os
import glob
import tensorflow as tf
from tensorflow.keras.optimizers import deserialize as deserialize_optimizer, serialize as serialize_optimizer, Optimizer, Adam

flags.DEFINE_string(
    'model', None, 'Name of the model (defined in moirai/architectures.py)')
flags.DEFINE_string(
    'decoder_model', None,
    'Name of the model (defined in moirai/architectures.py)')
flags.DEFINE_string('strategy', 'batch_triplets',
                    'Name of the strategy to use for training.')
flags.DEFINE_string(
    'config', None, 'Moirai configuration file specifying the'
    'inputs, validation sets, etc.')
flags.DEFINE_string(
    'checkpoint_file', None,
    'If provided, load weights from this checkpoint file before training.')
flags.DEFINE_boolean(
    'resume', None,
    'If True, look in checkpoint_dir for an appropriate checkpoint.')
flags.DEFINE_string(
    'checkpoint_dir', None,
    'If True, look in checkpoint_dir for an appropriate checkpoint.')

# moirai.common_flags
flags.declare_key_flag("epochs")
flags.declare_key_flag("batch_size")
flags.declare_key_flag("output_dir")

FLAGS = flags.FLAGS


def get_optional_config(config, name, raw=False):
    if name not in config:
        return None
    cfg = config[name]
    if raw:
        return cfg
    return json_dict_to_moirai_obj(cfg)


def main(argv):
    load_custom_configuration()

    print("Configuration file: '%s'" % FLAGS.config)
    print("Output directory: '%s'" % FLAGS.output_dir)

    if not tf.io.gfile.Exists(FLAGS.output_dir):
        tf.io.gfile.MakeDirs(FLAGS.output_dir)

    config = load_json_configuration_file(FLAGS.config)

    print(
        "Copying from '%s' to '%s'" %
        (FLAGS.config, os.path.join(FLAGS.output_dir, "input_configuration")))
    tf.io.gfile.Copy(
        FLAGS.config,
        os.path.join(FLAGS.output_dir, "input_configuration"),
        overwrite=True)

    reader = json_dict_to_moirai_obj(config['input'])
    optimizer = get_optional_config(config, "optimizer", raw=True)
    if not optimizer:
        optimizer = Adam(lr=.001)
    else:
        optimizer = deserialize_optimizer(optimizer)

    common_dataset = get_optional_config(config, "common_dataset")
    common_aug_dataset = get_optional_config(config,
                                             "common_augmented_dataset")
    dataset = get_optional_config(config, "dataset")
    augmented_dataset = get_optional_config(config, 'augmented_dataset')
    validation_dataset = get_optional_config(config, 'validation_dataset')

    augmentation = get_optional_config(config, "augmentation")
    filter = get_optional_config(config, 'filter')
    validation_reader = get_optional_config(config, 'validation_reader')

    decoder_losss = get_optional_config(
        config, 'decoder_losss', raw=True)
    visualizations = get_optional_config(config, 'visualizations', raw=True)

    if not dataset:
        dataset = common_dataset
    if not augmented_dataset:
        augmented_dataset = common_aug_dataset
    if not validation_dataset:
        validation_dataset = common_dataset

    if FLAGS.checkpoint_file:
        checkpoint_file = FLAGS.checkpoint_file
    elif FLAGS.resume and FLAGS.checkpoint_dir:
        path = os.path.join(FLAGS.checkpoint_dir, "keras_checkpoint*")
        print("Globbing %s" % path)
        list_of_files = glob.glob(path)
        if len(list_of_files):
            checkpoint_file = max(list_of_files, key=os.path.getctime)
        else:
            print("NO CHECKPOINT FOUND.")
            checkpoint_file = None
    else:
        checkpoint_file = None

    trainer.train(
        reader=reader,
        dataset=dataset,
        keras_checkpoint=checkpoint_file,
        filter=filter,
        augmented_dataset=augmented_dataset,
        augmentation=augmentation,
        validation_reader=validation_reader,
        validation_dataset=validation_dataset,
        epochs=FLAGS.epochs,
        prewarm_epochs=FLAGS.prewarm_epochs,
        model_name=FLAGS.model,
        decoder_model_name=FLAGS.decoder_model,
        decoder_losss=decoder_losss,
        visualizations=visualizations,
        num_gpus=FLAGS.num_gpus,
        optimizer=optimizer,
        strategy_name=FLAGS.strategy)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
    main(args)
