import random
import tensorflow as tf
from tqdm.auto import tqdm
from collections import defaultdict

from .samplers import Sampler


class MultiShotMemorySampler(Sampler):

    def __init__(self,
                 x,
                 y,
                 class_per_batch,
                 batch_size=32,
                 batch_per_epoch=1000,
                 augmenter=None,
                 scheduler=None,
                 warmup=-1):

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

    def get_examples(self, batch_id, num_classes, example_per_class):

        # select class at ramdom
        random.shuffle(self.class_list)
        class_list = self.class_list[:num_classes]

        # get example for each class
        idxs = []
        for class_id in class_list:
            class_idxs = self.index_per_class[class_id]
            random.shuffle(class_idxs)
            idxs.extend(class_idxs[:example_per_class])

        random.shuffle(idxs)
        batch_x = tf.gather(self.x, idxs[:self.batch_size])
        batch_y = tf.gather(self.y, idxs[:self.batch_size])

        return batch_x, batch_y


class SingleShotMemorySampler(Sampler):
    def __init__(self,
                 x,
                 augmenter,
                 class_per_batch,
                 batch_size=32,
                 batch_per_epoch=1000,
                 scheduler=None,
                 warmup=-1):

        super().__init__(class_per_batch,
                         batch_size=batch_size,
                         batch_per_epoch=batch_per_epoch,
                         augmenter=augmenter,
                         scheduler=scheduler,
                         warmup=warmup)
        self.x = x

        # each element is its own class
        self.idxs = list(range(len(x)))

    def get_examples(self, batch_id, num_classes, example_per_class):

        # select example at random as one elt == one class
        # the augmetnation is the one doing the extra example
        random.shuffle(self.idxs)
        y = self.idxs[:num_classes]
        x = [self.x[idx] for idx in y]

        return x, y
