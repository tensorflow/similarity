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

import abc
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
from tensorflow import Tensor
from tensorflow.keras.utils import Sequence

from tensorflow_similarity.augmenters import Augmenter

# Not currently used. Might be useful to allows gradual call of augmenter
Scheduler = Callable[[Any], Any]

# All basic types accepted by tf.keras.Model.fit(). This doesn't include tf.data
# datasets or keras generators.
Batch = Union[np.ndarray, List[np.ndarray], Tensor, List[Tensor], Mapping[str, Union[np.ndarray, Tensor]]]


class Sampler(Sequence, metaclass=abc.ABCMeta):
    def __init__(
        self,
        classes_per_batch: int,
        examples_per_class_per_batch: int = 2,
        num_augmentations_per_example: int = 0,
        steps_per_epoch: int = 1000,
        augmenter: Optional[Augmenter] = None,
        # scheduler: Optional[Scheduler] = None,
        warmup: int = 0,
    ) -> None:
        """Create a dataset sampler that ensure that each batch contains at
        least `examples_per_class_per_batch` examples of `classes_per_batch`
        classes. Sampling is needed as contrastive loss requires at least two
        examples of the same class present in a batch. The batch_size used
        during training will be equal to: `classes_per_batch` *
        `examples_per_class_per_batch` unless an `augmenter` that alters the
        number of examples returned is used. Then the batch_size is a function
        of how many augmented examples are returned by the `augmenter`.


        Args:
            classes_per_batch: Numbers of classes to include in a single batch

            examples_per_class_per_batch: how many examples of each class to
            use per batch. Defaults to 2.

            num_augmentations_per_example: how many augmented versions of an
            example will be produced by the augmenter. Defaults to 0.

            steps_per_epoch: How many steps/batches per epoch. Defaults to
            1000.

            augmenter: A function that takes a batch in and returns a batch
            out. Can alter the number of examples returned, which in turn can
            change the batch_size used. Defaults to None.

            warmup: Keep track of warmup epochs and let the augmenter know
            when the warmup is over by passing along with each batch a boolean
            `is_warmup`. See `self._get_examples()` Defaults to 0.
        """

        self.epoch = 0  # track epoch count
        self.classes_per_batch = classes_per_batch
        self.examples_per_class_per_batch = examples_per_class_per_batch
        self.num_augmentations_per_example = num_augmentations_per_example
        self.batch_size = classes_per_batch * examples_per_class_per_batch
        self.steps_per_epoch = steps_per_epoch
        self.augmenter = augmenter
        self.warmup = warmup
        self.is_warmup = True if warmup else False

        # Tell the users what to expect as they might be unsure what the batch
        # size will be
        print(
            f"\nThe initial batch size is {self.batch_size} "
            f"({self.classes_per_batch} classes * "
            f"{self.examples_per_class_per_batch} examples per class) with "
            f"{self.num_augmentations_per_example} augmenters"
        )

    @abc.abstractmethod
    def _get_examples(self, batch_id: int, num_classes: int, examples_per_class: int) -> Tuple[Batch, Batch]:
        """Get the set of examples that would be used to create a single batch.

        Notes:
         - before passing the batch data to TF, the sampler will call the
           augmenter function (if any) on the returned example.

         - A batch_size = num_classes * examples_per_class

         - This function must be defined in the subclass.

        Args:
            batch_id: id of the batch in the epoch.

            num_classes: How many class should be present in the examples.

            examples_per_class: How many example per class should be returned.

        Returns:
            x, y: batch of examples size `num_classes` * `examples_per_class`
        """

    # [Shared mechanics]
    def __len__(self) -> int:
        "Return the number of batch per epoch"
        return self.steps_per_epoch

    def on_epoch_end(self) -> None:
        "Keep track of warmup epochs"

        # if self.scheduler:
        #     # fixme scheduler idea
        #     self.scheduler(self.epoch)

        self.epoch += 1
        if self.is_warmup and (self.epoch >= self.warmup):
            print("Warmup complete")
            self.is_warmup = False

    def __getitem__(self, batch_id: int) -> Tuple[Batch, Batch]:
        return self.generate_batch(batch_id)

    def generate_batch(self, batch_id: int) -> Tuple[Batch, Batch]:
        """Generate a batch of data.


        Args:
            batch_id ([type]): [description]

        Returns:
            x, y: Batch
        """

        x, y = self._get_examples(batch_id, self.classes_per_batch, self.examples_per_class_per_batch)

        if self.augmenter:
            x, y = self.augmenter(x, y, self.num_augmentations_per_example, self.is_warmup)
        return x, y
