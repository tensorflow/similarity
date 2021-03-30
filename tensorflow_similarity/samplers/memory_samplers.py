import random
from collections import defaultdict
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow_similarity.types import Tensor
from tqdm.auto import tqdm

from .samplers import Augmenter, Sampler, Scheduler


class MultiShotMemorySampler(Sampler):

    def __init__(self,
                 x: Tensor,
                 y: Tensor,
                 class_per_batch: int,
                 batch_size: int = 32,
                 batch_per_epoch: int = 1000,
                 augmenter: Optional[Augmenter] = None,
                 scheduler: Optional[Scheduler] = None,
                 warmup: int = -1) -> None:

        super().__init__(class_per_batch,
                         batch_size=batch_size,
                         batch_per_epoch=batch_per_epoch,
                         augmenter=augmenter,
                         scheduler=scheduler,
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
                 batch_size: int = 32,
                 batch_per_epoch: int = 1000,
                 scheduler: Optional[Scheduler] = None,
                 warmup: int = -1) -> None:

        super().__init__(class_per_batch,
                         batch_size=batch_size,
                         batch_per_epoch=batch_per_epoch,
                         augmenter=augmenter,
                         scheduler=scheduler,
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
