# Copyright 2021 The TensorFlow Authors
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

import random
from collections import defaultdict
from typing import Optional, Tuple, TypeVar, Sequence

import tensorflow as tf
from tensorflow_similarity.types import FloatTensor, IntTensor, Tensor
from tqdm.auto import tqdm

from .samplers import Augmenter, Sampler
from .utils import select_examples

T = TypeVar('T', FloatTensor, IntTensor)


class MultiShotMemorySampler(Sampler):
    # FIXME: typing x, y is hard.
    def __init__(self,
                 x,
                 y,
                 classes_per_batch: int = 2,
                 examples_per_class_per_batch: int = 2,
                 steps_per_epoch: int = 1000,
                 class_list: Sequence[int] = None,
                 total_examples_per_class: int = None,
                 augmenter: Optional[Augmenter] = None,
                 warmup: int = -1):
        """Create a Multishot in memory sampler that ensures that each batch is
        well balanced. That is, each batch aims to contain
        `example_per_class` examples of `class_per_batch`
        classes.

        The `batch_size` used during training will be equal to:
        `class_per_batch * example_per_class` unless an `augmenter` that alters
        the number of examples returned is used. Then the batch_size is a
        function of how many augmented examples are returned by
        the `augmenter`.


        Multishot samplers are to be used when you have multiple
        examples for the same class. If this is not the case, then see
        the [SingleShotMemorySampler()](single_memory.md) for using single
        example with augmentation.

        Memory samplers are good for datasets that fit in memory. If you have
        larger ones that needs to sample from disk then use the
        [TFRecordDatasetSampler()](tfdataset_sampler.md)


        Args:
            x: examples.

            y: labels.

            class_per_batch: Numbers of distinct class to include in a
            single batch

            examples_per_class_per_batch: How many example of each class
            to use per batch. Defaults to 2.

            steps_per_epoch: How many steps/batches per epoch.
            Defaults to 1000.

            class_list: Filter the list of examples to only keep those who
            belong to the supplied class list.

            total_examples_per_class: Restrict the number of examples for EACH
            class to total_examples_per_class if set. If not set, all the
            available examples are selected. Defaults to None - no selection.


            augmenter: A function that takes a batch in and return a batch out.
            Can alters the number of examples returned which in turn change the
            batch_size used. Defaults to None.

            warmup: Keep track of warmup epochs and let the augmenter knows
            when the warmup is over by passing along with each batch data a
            boolean `is_warmup`. See `self.get_examples()` Defaults to 0.
        """

        super().__init__(
            classes_per_batch,
            examples_per_class_per_batch=examples_per_class_per_batch,  # noqa
            steps_per_epoch=steps_per_epoch,
            augmenter=augmenter,
            warmup=warmup)

        # precompute information we need
        if not class_list:
            self.class_list = list(set([int(e) for e in y]))
        else:
            # dedup in case user mess up and cast in case its a tensor
            self.class_list = list(set([int(c) for c in class_list]))

        if classes_per_batch > len(self.class_list):
            raise ValueError("the value of class_per_batch must be <= to the\
                                number of existing class in the dataset")

        # filter
        x, y = select_examples(x,
                               y,
                               class_list=self.class_list,
                               num_examples_per_class=total_examples_per_class)

        # assign after potential selection
        self._x = x
        self._y = y

        # we need to reindex here  as the numbers of samples might have
        # changed due to the filtering
        # In this sampler, contrary to file based one, the pool of examples
        # wont' change so we can optimize by doing this step in the constructor
        self.index_per_class = defaultdict(list)
        cls = [int(c) for c in y]  # need to cast as  tensor lookup are slowww
        for idx in tqdm(range(len(x)), desc='indexing classes'):
            cl = cls[idx]
            self.index_per_class[cl].append(idx)

    def get_examples(self,
                     batch_id: int,
                     num_classes: int,
                     examples_per_class: int) -> Tuple[Tensor, Tensor]:
        # select class at ramdom
        class_list = random.sample(self.class_list, k=num_classes)

        # get example for each class
        idxs = []
        for class_id in class_list:
            class_idxs = self.index_per_class[class_id]
            idxs.extend(random.choices(class_idxs, k=examples_per_class))

        random.shuffle(idxs)
        idxs_slice = tf.constant(idxs[:self.batch_size])

        with tf.device("/cpu:0"):
            batch_x = tf.gather(self._x, indices=idxs_slice)
            batch_y = tf.gather(self._y, indices=idxs_slice)

        return batch_x, batch_y

    def get_slice(self,
                  begin: int = 0,
                  size: int = -1) -> Tuple[FloatTensor, IntTensor]:
        """Extracts a slice over both the x and y tensors.

        This method extracts a slice of size `size` over the first dimension of
        both the x and y tensors starting at the index specified by `begin`.

        The value of `begin + size` must be less than `self.num_examples`.

        Args:
            begin: The starting index.
            size: The size of the slice.

        Returns:
            A Tuple of FloatTensor and IntTensor
        """
        slice_x: FloatTensor = self._get_slice(self._x, begin, size)
        slice_y: IntTensor = self._get_slice(self._y, begin, size)

        return slice_x, slice_y

    def _get_slice(self,
                   input_: T,
                   begin: int,
                   size: int) -> T:
        b = [0] * len(tf.shape(input_))
        b[0] = begin
        s = [-1] * len(tf.shape(input_))
        s[0] = size

        slice_: T = tf.slice(input_, b, s)
        return slice_

    @property
    def num_examples(self) -> int:
        return len(self._x)

    @property
    def example_shape(self):
        return self._x[0].shape


class SingleShotMemorySampler(Sampler):
    def __init__(self,
                 x: FloatTensor,
                 augmenter: Augmenter,
                 class_per_batch: int,
                 steps_per_epoch: int = 1000,
                 warmup: int = -1) -> None:
        """Create a single shot in memory sampler.

        The `batch_size` used during training will be equal to:
        `class_per_batch * example_per_class` unless the `augmenter`
        used to create similar looking examples have a different regime.
        In that case the batch_size would be a function of how many augmented
        examples are returned by the `augmenter` function.

        Single shot samplers are used when you have a single example per class
        and solely rely on data augmentation to generate similar looking
        examples.

        If you do have multiple examples per class, you should use the
        [MultiShotMemorySampler()](multishot_memory.md).

        Memory samplers are good for datasets that fit in memory. If you have
        larger ones that needs to sample from disk then use the
        [TFRecordDatasetSampler()](tfdataset_sampler.md).


        Args:

            x: Input data. The sampler assumes that each element of X is from a
            distinct class.

            augmenter: A function that takes a batch of single examples and
            return a batch out with additional examples per class.

            steps_per_epoch: How many steps/batch per epoch. Defaults to 1000.

            class_per_batch: effectively the number of element to pass to the
            augmenter for each batch request in the single shot setting.

            warmup: Keep track of warmup epochs and let the augmenter knows
            when the warmup is over by passing along with each batch data a
            boolean `is_warmup`. See `self.get_examples()` Defaults to 0.
        """

        super().__init__(class_per_batch,
                         steps_per_epoch=steps_per_epoch,
                         augmenter=augmenter,
                         warmup=warmup)
        self.x = x
        self.num_elts = len(x)

    def get_examples(self, batch_id: int, num_classes: int,
                     example_per_class: int) -> Tuple[Tensor, Tensor]:
        _ = batch_id
        _ = example_per_class

        # note: we draw at random the class so the sampler can scale up to
        # millions of points. Shuffling array is simply too slow
        idxs = tf.random.uniform((num_classes, ),
                                 minval=0,
                                 maxval=self.num_elts,
                                 dtype='int32')
        # ! don't cast data as different model use different type.
        y = tf.constant([int(i) for i in idxs])
        x = tf.constant([self.x[idx] for idx in y])

        return x, y
