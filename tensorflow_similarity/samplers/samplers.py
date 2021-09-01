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
from typing import Any, Callable, Optional, Tuple

from tensorflow.keras.utils import Sequence
from tensorflow_similarity.types import Tensor

# An Augmenter is a Map TensorLike -> TensorLike. The function must
# implement the following signature:
#
# Args:
#   x: Feature Tensors
#   y: Label Tensors
#   examples_per_class: The number of examples per class.
#   is_warmup: If True, the sampler is still in warmup phase.
# Returns:
#   A Tuple containing the transformed x and y tensors.
Augmenter = (Callable[[Tensor, Tensor, int, bool], Tuple[Tensor, Tensor]])

# Not currently used. Might be useful to allows gradual call of augmenter
Scheduler = Callable[[Any], Any]


class Sampler(Sequence, metaclass=abc.ABCMeta):
    def __init__(
            self,
            classes_per_batch: int,
            examples_per_class_per_batch: int = 2,
            steps_per_epoch: int = 1000,
            augmenter: Optional[Augmenter] = None,
            # scheduler: Optional[Scheduler] = None,
            warmup: int = 0) -> None:
        """Create a dataset sampler that ensure that each batch contains at
        least `example_per_class` examples of `class_per_batch`
        classes. Sampling is needed as contrastive loss requires at
        least two examples of the same class present in a batch.
        The batch_size used during training will be equal to:
        `class_per_batch * example_per_class` unless an `augmenter` that alters
        the number of examples returned is used. Then the batch_size is a
        function of how many augmented examples are returned by
        the `augmenter`.


        Args:
            class_per_batch: Numbers of class to include in a single batch
            example_per_class: How many example to include for each class per
            batch.

            example_per_class: how many example of each class to use per batch.
            Defaults to 2.

            steps_per_epoch: How many steps/batches per epoch. Defaults to
            1000.

            augmenter: A function that takes a batch in and return a batch out.
            Can alters the number of examples returned which in turn change the
            batch_size used. Defaults to None.

            warmup: Keep track of warmup epochs and let the augmenter knows
            when the warmup is over by passing along with each batch data a
            boolean `is_warmup`. See `self.get_examples()` Defaults to 0.
        """

        self.epoch = 0  # track epoch count
        self.classes_per_batch = classes_per_batch
        self.examples_per_class_per_batch = examples_per_class_per_batch
        self.batch_size = classes_per_batch * examples_per_class_per_batch
        self.steps_per_epoch = steps_per_epoch
        self.augmenter = augmenter
        self.warmup = warmup
        self.is_warmup = True if warmup else False

        # Tell the users what to expect as they might be unsure what the batch
        # size will be
        print("\nBatch size is %d (%d class X %d example per class "
              "pre-augmentation" % (self.batch_size, self.classes_per_batch,
                                    self.examples_per_class_per_batch))

    @abc.abstractmethod
    def get_examples(
        self, batch_id: int, num_classes: int, example_per_class: int
    ) -> Tuple[Any, Any]:  # FIXME: Tensor type cause errors.
        """Get the set of examples that would be used to create a single batch.

        Notes:
         - before passing the batch data to TF, the sampler will call the
           augmenter function (if any) on the returned example.

         - A batch_size = num_classes * example_per_class

         - This function must be defined in the subclass.

        Args:
            batch_id: id of the batch in the epoch.
            num_classes: How many class should be present in the examples.
            example_per_class: How many example per class should be returned.

        Returns:
            x, y: batch of examples made of `num_classes` * `example_per_class`
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

    def __getitem__(self, batch_id: int) -> Tuple[Tensor, Tensor]:
        return self.generate_batch(batch_id)

    def generate_batch(self, batch_id: int) -> Tuple[Tensor, Tensor]:
        """Generate a batch of data.


        Args:
            batch_id ([type]): [description]

        Returns:
            x, y: batch
        """

        x, y = self.get_examples(batch_id, self.classes_per_batch,
                                 self.examples_per_class_per_batch)

        # strip examples if needed. This might happen due to rounding
        if len(x) != self.batch_size:
            x = x[:self.batch_size]
            y = y[:self.batch_size]

        if self.augmenter:
            x, y = self.augmenter(x, y, self.examples_per_class_per_batch,
                                  self.is_warmup)
        return x, y
