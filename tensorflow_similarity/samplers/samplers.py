import abc
import math
from typing import Any, Callable, Optional, Tuple

from tensorflow.keras.utils import Sequence
from tensorflow.types import experimental as tf_types

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
Augmenter = (Callable[[tf_types.TensorLike, tf_types.TensorLike, int, bool],
                      Tuple[tf_types.TensorLike, tf_types.TensorLike]])

# Not currently used.
Scheduler = Callable[[Any], Any]


class Sampler(Sequence, metaclass=abc.ABCMeta):

    def __init__(self,
                 class_per_batch: int,
                 batch_size: int = 32,
                 batch_per_epoch: int = 1000,
                 augmenter: Optional[Augmenter] = None,
                 scheduler: Optional[Scheduler] = None,
                 warmup: int = 0) -> None:

        self.epoch = 0  # track epoch count
        self.class_per_batch = class_per_batch
        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch
        self.augmenter = augmenter
        self.scheduler = scheduler
        self.warmup = warmup
        self.is_warmup = True if warmup else False

    @abc.abstractmethod
    def get_examples(self,
                     batch_id: int,
                     num_classes: int,
                     example_per_class: int
                     ) -> Tuple[tf_types.TensorLike, tf_types.TensorLike]:
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
        """
        raise NotImplementedError('must be implemented by subclass')

    # [Shared mechanics]
    def __len__(self) -> int:
        return self.batch_per_epoch

    def on_epoch_end(self) -> None:
        # # scheduler -> batch_size, class_per_batch?
        # if self.scheduler:
        #     # fixme scheduler idea
        #     self.scheduler(self.epoch)

        self.epoch += 1
        if self.epoch == self.warmup and self.is_warmup:
            print("Warmup complete")
            self.is_warmup = False

    def __getitem__(self,
                    batch_id: int
                    ) -> Tuple[tf_types.TensorLike, tf_types.TensorLike]:

        return self.generate_batch(batch_id)

    def generate_batch(self,
                       batch_id: int
                       ) -> Tuple[tf_types.TensorLike, tf_types.TensorLike]:

        example_per_class = math.ceil(self.batch_size / self.class_per_batch)
        # ! can't have less than 2 example per class in a batch
        example_per_class = max(example_per_class, 2)

        x, y = self.get_examples(batch_id, self.class_per_batch,
                                 example_per_class)

        # strip an example if need to be. This might happen due to rounding
        if len(x) != self.batch_size:
            x = x[:self.batch_size]
            y = y[:self.batch_size]

        if self.augmenter:
            x, y = self.augmenter(x, y, example_per_class, self.is_warmup)
        return x, y
