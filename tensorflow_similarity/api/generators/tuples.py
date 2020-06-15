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
import collections
import itertools
import json
import math
import os
import random
import time
import traceback
from datetime import datetime

import numpy as np
import numpy.random as rng
import six
import tensorflow as tf
from absl import app, flags
from PIL import Image, ImageDraw
from tensorboard.plugins.text.summary_v2 import text as text_summary
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2

from tensorflow_similarity.api.engine.augmentation import IdentityAugmentation
from tensorflow_similarity.api.engine.blackboard import Blackboard
from tensorflow_similarity.api.engine.generator import Batch, Element, Generator
from tensorflow_similarity.api.engine.logging import get_logger
from tensorflow_similarity.api.engine.preprocessing import IdentityPreprocessing
from tensorflow_similarity.dataset import Dataset, Datum
from tensorflow_similarity.utils.config_utils import (deserialize_moirai_object,
                                       register_custom_object,
                                       value_or_callable)

FLAGS = flags.FLAGS


class ElementSpec(object):
    def __init__(self, tower_name, should_augment=False):
        self.tower_name = tower_name
        self.should_augment = should_augment

    def __repr__(self):
        return ("%s - should%saugment" %
                (self.tower_name, " " if self.should_augment else " not "))


class TupleSpec(object):
    def __init__(self, elements=[], groups=[]):

        self.groups = groups
        self.tower_map = {}
        self.elements = []

        for e in elements:
            e = deserialize_moirai_object(e)
            self.elements.append(e)
        self.elements.sort(key=lambda x: x.tower_name)

        self.positive_pairs = []
        self.negative_pairs = []
        self._generate_pairs(groups)

    def __str__(self):
        out = []
        for e in self.elements:
            out.append(str(e))
        for g in self.groups:
            out.append(str(g))

        return ("\n".join(out))

    def _generate_pairs(self, groups):
        for group in groups:
            intragroup_pairs = list(itertools.combinations(group, 2))
            self.positive_pairs.extend(intragroup_pairs)

        for group_pair in itertools.combinations(groups, 2):
            for l_item in group_pair[0]:
                for r_item in group_pair[1]:
                    self.negative_pairs.append((l_item, r_item))

    def get_tower_names(self):
        return [e.tower_name for e in self.elements]

    def should_augment(self):
        return [e.should_augment for e in self.elements]

    def distribute_classes(self, classes):
        """Input is the set of class IDs. This is combined with the sorted
        tower order to create an array of class IDs in the order the
        model will expect them.  E.g. for quads, you have:
        "anchor, n1, n2, pos" - however, anchor and pos share the 0th class.
        """

        element_to_class = {}

        for group, cls in zip(self.groups, classes):
            for tower_name in group:
                element_to_class[tower_name] = cls

        sorted_values = [
            element_to_class[name] for name in self.get_tower_names()
        ]
        return sorted_values


class NewTupleGenerator(Generator):
    def __init__(self,
                 examples,
                 labels,
                 tuple_spec,
                 pair_filter=None,
                 preprocessing=None,
                 augmentation=None,
                 max_filter_fail_streak=1000,
                 batch_size=128,
                 step_size_multiplier=1,
                 generate_autoencoder_labels=False,
                 deterministic_outputs=False,
                 log_dir=None,
                 visualize_text_fields=[],
                 visualize_img_fields=[],
                 visualize_img_denormalization=[],
                 debug=0):

        self.log_dir = log_dir
        self.has_written_visualizations = False
        self.visualize_text_fields = visualize_text_fields
        self.visualize_img_fields = visualize_img_fields
        self.visualize_img_denormalization = visualize_img_denormalization
        self.deterministic_outputs = deterministic_outputs
        self.debug = debug
        self.meta_features = ["idx", "dataset", "label", "generation", "group"]

        self.tuple_spec = tuple_spec
        self.tower_names = self.tuple_spec.get_tower_names()
        self.generate_autoencoder_labels = generate_autoencoder_labels
        self.examples = examples
        self.labels = labels
        self.integerized_labels = []
        self.class_map = []
        self.num_classes = 0
        self.augmentation = deserialize_moirai_object(
            augmentation) or IdentityAugmentation()

        self.preprocessing = deserialize_moirai_object(
            preprocessing) or IdentityPreprocessing()

        self.pair_filter = deserialize_moirai_object(pair_filter)
        self.max_filter_fail_streak = deserialize_moirai_object(
            max_filter_fail_streak)
        self.batch_size = deserialize_moirai_object(batch_size)
        self.step_size_multiplier = deserialize_moirai_object(
            step_size_multiplier)
        self.batches_per_epoch = None
        self.batch_idx = 0

        self._index()

    def get_tower_names(self):
        return self.tuple_spec.get_tower_names()

    def _index(self):
        class_dict = {}
        class_map = []
        for i, label in enumerate(self.labels):
            if label not in class_dict:
                class_dict[label] = len(class_dict)
                class_map.append([])
            label_id = class_dict[label]
            class_map[label_id].append(i)
            self.integerized_labels.append(label_id)
        self.class_map = class_map
        self.num_classes = len(self.class_map)

    def filter_tuple(self, elements_by_tower_name):
        if not self.pair_filter:
            return True

        for left, right in self.tuple_spec.positive_pairs:
            if not self.pair_filter.keep_positive(
                    elements_by_tower_name[left],
                    elements_by_tower_name[right]):
                return False

        for left, right in self.tuple_spec.negative_pairs:
            if not self.pair_filter.keep_negative(
                    elements_by_tower_name[left],
                    elements_by_tower_name[right]):
                return False
        return True

    def __len__(self):
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch

        first_key = list(self.examples.keys())[0]
        num_examples = len(self.examples[first_key])
        self.batches_per_epoch = int(
            math.ceil(self.step_size_multiplier *
                      (float(num_examples) / self.batch_size)))

        return self.batches_per_epoch

    def seed(self, batch_id):
        """Sets a seed for the given batch, based on the PID of the process,
        the batch id, and the current time. This avoids an issue where several
        generators are forked with the same RNG state, resulting in the
        generators all generating the same sequences."""

        if self.deterministic_outputs:
            seed = batch_id
        else:
            pid = os.getpid()
            time_micros = datetime.now().microsecond
            seed = (time_micros * pid + batch_id) % 2000000000
        rng.seed(seed)
        np.random.seed(seed)
        return seed

    def _retrying_random_sample(self):
        fail_streak = 0
        values = None
        while values is None and fail_streak <= self.max_filter_fail_streak:
            values = self.random_sample()
            fail_streak += 1

        if values is None:
            raise ValueError(
                "Could not find a suitable tuple after %d consecutive attempts."
                % self.max_filter_fail_streak)
        return values

    def __getitem__(self, batch_id):
        batch = self.get_batch(batch_id)
        return batch.values, batch.labels

    def get_batch(self, batch_id):
        self.seed(batch_id)
        self.on_generate_batch_start(batch_id)

        inputs = collections.defaultdict(list)
        preprocessed_inputs = collections.defaultdict(list)
        augmented_inputs = collections.defaultdict(list)
        targets = collections.defaultdict(list)

        try:
            for _ in range(self.batch_size):
                values = self._retrying_random_sample()

                for tower_name, tuple_element in six.iteritems(values):
                    idx = tuple_element.index
                    raw_features = tuple_element.raw_feature_dictionary
                    pp_raw_features = tuple_element.preprocessed_dictionary
                    aug_features = tuple_element.feature_dictionary

                    for k, v in six.iteritems(raw_features):
                        inputs["%s_%s" % (tower_name, k)].append(v)

                    for k, v in six.iteritems(pp_raw_features):
                        preprocessed_inputs["%s_%s" %
                                            (tower_name, k)].append(v)

                    for k, v in six.iteritems(aug_features):
                        augmented_inputs["%s_%s" % (tower_name, k)].append(v)

                    augmented_inputs["%s_idx" % tower_name].append(idx)
                    targets["%s_idx_out" % tower_name].append(idx)

            batch = Batch()
            batch.add_raw_features("main", inputs)
            batch.add_preprocessed_raw_features("main", preprocessed_inputs)
            batch.add_features("main", augmented_inputs)
            batch.add_labels("main", targets)

            if not self.has_written_visualizations:
                if self.log_dir and batch_id:
                    self.visualize_batch(batch_id, batch)
                    self.has_written_visualizations = True

            self.on_generate_batch_end(batch_id)
            return batch

        except Exception:
            traceback.print_exc()
            exit(1)

    def visualize_text(self, writer, batch_id, batch, field):
        text_array = []

        # Header row
        text_array.append(["**%s**" % tower for tower in self.tower_names])

        field = field.strip()

        arrays = []
        for tower in self.tower_names:
            item = "%s_%s" % (tower, field)
            arrays.append(batch.values[item])

        for tuple in zip(*arrays):
            text_array.append([str(x) for x in tuple])

        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                with writer.as_default():
                    text_summary(name="generated_%s" % field,
                                 data=tf.convert_to_tensor(
                                     text_array, dtype=tf.string),
                                 step=batch_id)

    def visualize_img(self, writer, batch_id, batch, field, denormalization):

        img0 = np.shape(batch.values["%s_%s" % (self.tower_names[0], field)])
        one_img_shape = img0[1:]
        w = img0[0] + 2
        h = img0[1] + 2
        channels = img0[2]

        width = w * len(self.tower_names)
        height = h * (self.batch_size + 1)

        final_img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(final_img)

        for tidx, tower in enumerate(self.tower_names):
            draw_position = (tidx*w + 1, int(h/2))
            draw.text(draw_position, tower, fill=(255, 255, 255))

            item = "%s_%s" % (tower, field)
            raw_imgs = batch.values[item]

            for idx, raw_img in enumerate(raw_imgs):
                if denormalization:
                    denormed = denormalization(raw_img)
                else:
                    denormed = raw_img

                img = Image.fromarray(denormed)
                final_img.paste(img, (w * tidx, h * (idx + 1)))

        final_img = np.expand_dims(np.array(final_img), 0)
        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                with writer.as_default():
                    summary_ops_v2.image(name="generator_%s" % field,
                                         tensor=tf.convert_to_tensor(
                                             final_img),
                                         family="visualizations", step=batch_id)

    def visualize_batch(self, batch_id, batch):
        log_dir = os.path.join(self.log_dir, "generator")

        writer = summary_ops_v2.create_file_writer_v2(log_dir)

        if self.visualize_text_fields:
            for field in self.visualize_text_fields:
                self.visualize_text(writer, batch_id, batch, field)

        if self.visualize_img_fields:
            denorms = []
            if isinstance(self.visualize_img_denormalization, list):
                denorms = self.visualize_img_denormalization
            else:
                for _ in self.visualize_img_fields:
                    denorms.append(self.visualize_img_denormalization)

            for field, denorm in zip(self.visualize_img_fields, denorms):
                self.visualize_img(writer, batch_id, batch, field, denorm)

        writer.close()

    def indices_to_examples(self, idxs, should_augment):
        output = {}

        # self.examples is dict of name -> arrays
        # we need a list of dictionaries of features (one per example)
        # TODO - probably worth preprocessing this at the beginning.
        examples = []
        for idx in idxs:
            example = {}
            for k, v in six.iteritems(self.examples):
                example[k] = v[idx]
            examples.append(example)
        labels = [self.labels[idx] for idx in idxs]

        for idx, tower_name, raw_feature_dictionary, augment, label in zip(
                idxs, self.tower_names, examples, should_augment, labels):

            aug_dictionary = self.augmentation.augment(raw_feature_dictionary)
            preprocessed_dictionary = self.preprocessing.preprocess(
                raw_feature_dictionary)
            if augment:
                aug_dictionary = self.preprocessing.preprocess(aug_dictionary)
            else:
                aug_dictionary = preprocessed_dictionary

            el = Element(
                index=idx,
                feature_dictionary=aug_dictionary,
                preprocessed_dictionary=preprocessed_dictionary,
                raw_feature_dictionary=raw_feature_dictionary,
                label_id=label,
                augmented=1 if augment else 0)

            output[tower_name] = el
        return output

    def small_random_sample(self, max_id, N=3):
        """Randomly select N class indices from the dataset, without
        replacement.

        Note that N should typically be small, relative to the size of your
        dataset. If for some reason you need a large N, you may be better
        off with a `np.random.randint(self.num_classes, N, replace=False)`

        Keyword Arguments:
            N {int} -- Number of classes to randomly select (default: {3})

        Returns:
            [list(int)] -- A list of class indices selected.
        """

        class_indexes = []

        # Effectively: we choose the first element at random from #classes.
        # Then, we choose the next element by choosing a random number from
        # 0 - #classes-1, and manually skip over the elements we've already
        # chosen.
        for i in range(N):
            idx = rng.randint(0, max_id - i)

            if not len(class_indexes):
                class_indexes.append(idx)
            else:
                previous_insert_point = 0
                updated_insert_point = bisect.bisect_right(class_indexes, idx)

                new_idx = idx + updated_insert_point

                while updated_insert_point != previous_insert_point:
                    previous_insert_point = updated_insert_point
                    updated_insert_point = bisect.bisect_right(
                        class_indexes, new_idx)
                    new_idx = idx + updated_insert_point

                bisect.insort(class_indexes, new_idx)
        np.random.shuffle(class_indexes)
        return class_indexes

    def random_sample(self):
        num_groups = len(self.tuple_spec.groups)

        # E.g. for quads, selects a "pos/anchor", "neg1", and "neg2" class.
        elements_by_class = self.small_random_sample(self.num_classes,
                                                     num_groups)

        if self.debug > 1:
            debug("EBC" + str(elements_by_class))

        # E.g. for quads, transforms the above into
        # [anchor, neg1, neg2, pos] = [
        #     class_idxs[0], class_idxs[1], class_idxs[2], class_idxs[0]]
        tower_ordered_classes = self.tuple_spec.distribute_classes(
            elements_by_class)

        if self.debug > 1:
            debug("TOC" + str(tower_ordered_classes))

        # For each class/tower, look up the set of indices for the class.
        choices_per_tower = [self.class_map[c] for c in tower_ordered_classes]

        if self.debug > 1:
            debug("CpT" + str(choices_per_tower))

        idxs = [np.random.choice(c) for c in choices_per_tower]

        if self.debug > 1:
            debug("I" + str(idxs))

        output = self.indices_to_examples(
            idxs, should_augment=self.tuple_spec.should_augment())

        if self.debug > 1:
            debug("Final:", str(output))

        if not self.filter_tuple(output):
            return None

        return output

    def on_generate_batch_start(self, batch_id):
        pass

    def on_generate_batch_end(self, batch_id):
        pass


class QuadrupletGenerator(NewTupleGenerator):
    def __init__(self, examples, labels, **kwargs):

        anchor = ElementSpec("anchor", should_augment=False)
        neg1 = ElementSpec("neg1", should_augment=True)
        neg2 = ElementSpec("neg2", should_augment=True)
        pos = ElementSpec("pos", should_augment=True)

        tuple_spec = TupleSpec(
            elements=[anchor, neg1, neg2, pos],
            groups=[
                # pos == anchor, pos != neg1, neg1 != neg2, pos != neg2
                ["pos", "anchor"],
                ["neg1"],
                ["neg2"]
            ])

        super(QuadrupletGenerator, self).__init__(examples, labels, tuple_spec,
                                                  **kwargs)


class TripletGenerator(NewTupleGenerator):
    def __init__(self, examples, labels, **kwargs):

        anchor = ElementSpec("anchor", should_augment=False)
        neg = ElementSpec("neg", should_augment=True)
        pos = ElementSpec("pos", should_augment=True)

        tuple_spec = TupleSpec(
            elements=[anchor, neg, pos],
            groups=[
                # pos == anchor, pos != neg1, neg1 != neg2, pos != neg2
                ["pos", "anchor"],
                ["neg"]
            ])

        super(TripletGenerator, self).__init__(examples, labels, tuple_spec,
                                               **kwargs)


class BlackboardHardMining(NewTupleGenerator):
    def __init__(self,
                 basic_generator=None,
                 hard_mining_directory=None,
                 recycle_frequency=.25,
                 debug=0,
                 **kwargs):
        self.basic_generator = deserialize_moirai_object(basic_generator)

        super(BlackboardHardMining, self).__init__(
            examples=[],
            labels=[],
            tuple_spec=basic_generator.tuple_spec,
            **kwargs)

        self.basic_generator = basic_generator
        self.tuple_spec = basic_generator.tuple_spec

        self.blackboard = Blackboard(
            os.path.join(hard_mining_directory, "blackboard.data"),
            basic_generator.get_tower_names())
        self.blackboard_data = None
        self.known_hard_tuples_reuse_count = 0
        self.new_tuple_count = 0
        self.batches = 0
        self.recycle_frequency = recycle_frequency
        self.debug = debug

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                "basic_generator": self.basic_generator.get_config(),
                "recycle_frequency": self.recycle_frequency,
                "hard_mining_directory": self.hard_mining_directory
            }
        }

    def __len__(self):
        return len(self.basic_generator)

    def steps_per_epoch(self):
        return self.basic_generator.steps_per_epoch()

    def random_sample(self):
        # Grab the reference to the blackboard data, to avoid issues with
        # the data changing out from under us.
        blackboard_data = self.blackboard_data

        if (not blackboard_data.empty()
                and rng.uniform() < self.recycle_frequency):
            idxs = blackboard_data.sample()
            if idxs is not None:
                self.known_hard_tuples_reuse_count += 1

                return self.basic_generator.indices_to_examples(
                    idxs, should_augment=self.tuple_spec.should_augment())

        self.new_tuple_count += 1
        out = self.basic_generator.random_sample()
        return out

    def on_generate_batch_start(self, batch_id):
        self.blackboard_data = self.blackboard.get()

    def on_generate_batch_end(self, batch_id):
        if batch_id % 10 == 0:
            total = self.known_hard_tuples_reuse_count + self.new_tuple_count
            if self.debug >= 1:
                get_logger().info(
                    "Hard tuple frequency: %f" %
                    (float(self.known_hard_tuples_reuse_count) / total))

            self.known_hard_tuples_reuse_count = 0
            self.new_tuple_count = 0


class HardTripletGenerator(BlackboardHardMining):
    def __init__(self,
                 examples,
                 labels,
                 hard_mining_directory=None,
                 recycle_frequency=.5,
                 **kwargs):
        super(HardTripletGenerator, self).__init__(
            basic_generator=TripletGenerator(examples, labels, **kwargs),
            hard_mining_directory=hard_mining_directory,
            recycle_frequency=recycle_frequency,
            **kwargs)


class HardQuadrupletGenerator(BlackboardHardMining):
    def __init__(self,
                 examples,
                 labels,
                 hard_mining_directory=None,
                 recycle_frequency=.5,
                 **kwargs):
        super(HardQuadrupletGenerator, self).__init__(
            basic_generator=QuadrupletGenerator(examples, labels, **kwargs),
            hard_mining_directory=hard_mining_directory,
            recycle_frequency=recycle_frequency,
            **kwargs)
