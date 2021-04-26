import random
from collections import defaultdict
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow_similarity.types import Tensor
from tqdm.auto import tqdm

from .samplers import Augmenter, Sampler


class MultiShotMemorySampler(Sampler):

    def __init__(self,
                 x: Tensor,
                 y: Tensor,
                 class_per_batch: int,
                 example_per_class: int = 2,
                 batch_per_epoch: int = 1000,
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


        Multishot samplers are to be used when you have multiples
        examples for the same class. If you don't then see
        the [SingleShotMemorySampler()](single_memory.md) for using single
        example with augmentation.

        Memory samplers are good for datasets that fit in memory. If you have
        larger ones that needs to sample from disk then use the
        [TFRecordDatasetSampler()](tfdataset_sampler.md)


        Args:
            x: examples.

            y: labels.

            class_per_batch: Numbers of class to include in a single batch
            example_per_class: How many example to include for each class per
            batch.

            example_per_class: how many example of each class to use per batch.
            Defaults to 2.

            batch_per_epoch: How many batch per epoch. Defaults to 1000.

            augmenter: A function that takes a batch in and return a batch out.
            Can alters the number of examples returned which in turn change the
            batch_size used. Defaults to None.

            warmup: Keep track of warmup epochs and let the augmenter knows
            when the warmup is over by passing along with each batch data a
            boolean `is_warmup`. See `self.get_examples()` Defaults to 0.
        """

        super().__init__(class_per_batch,
                         example_per_class=example_per_class,
                         batch_per_epoch=batch_per_epoch,
                         augmenter=augmenter,
                         warmup=warmup)
        self.x = x
        self.y = y

        # precompute information we need
        class_list, _ = tf.unique(y)
        self.class_list = [int(e) for e in class_list]

        # Mapping class to idx
        # In this sampler, contrary to file based one, the pool of examples
        # wont' change so we can optimize by doing this step in the constructor
        self.index_per_class = defaultdict(list)
        for idx in tqdm(range(len(x)), desc='indexing classes'):
            cl = int(y[idx])  # need to cast tensor
            self.index_per_class[cl].append(idx)

    def get_examples(self,
                     batch_id: int,
                     num_classes: int,
                     example_per_class: int
                     ) -> Tuple[Tensor, Tensor]:

        # select class at ramdom
        class_list = random.sample(self.class_list, k=num_classes)

        # get example for each class
        idxs = []
        for class_id in class_list:
            class_idxs = self.index_per_class[class_id]
            idxs.extend(random.choices(class_idxs, k=example_per_class))

        random.shuffle(idxs)
        batch_x = tf.gather(self.x, idxs[:self.batch_size])
        batch_y = tf.gather(self.y, idxs[:self.batch_size])

        return batch_x, batch_y


class SingleShotMemorySampler(Sampler):
    def __init__(self,
                 x: Tensor,
                 augmenter: Augmenter,
                 class_per_batch: int,
                 batch_per_epoch: int = 1000,
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

            batch_per_epoch: How many batch per epoch. Defaults to 1000.

            class_per_batch: effectively the number of element to pass to the
            augmenter for each batch request in the single shot setting.

            warmup: Keep track of warmup epochs and let the augmenter knows
            when the warmup is over by passing along with each batch data a
            boolean `is_warmup`. See `self.get_examples()` Defaults to 0.
        """

        super().__init__(class_per_batch,
                         batch_per_epoch=batch_per_epoch,
                         augmenter=augmenter,
                         warmup=warmup)
        self.x = x
        self.num_elts = len(x)

    def get_examples(self,
                     batch_id: int,
                     num_classes: int,
                     example_per_class: int
                     ) -> Tuple[Tensor, Tensor]:
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
