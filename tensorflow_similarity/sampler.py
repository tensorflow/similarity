import random
import tensorflow as tf
from tqdm.auto import tqdm
from tensorflow.keras.utils import Sequence
from collections import defaultdict


def select_examples(x, y, class_list, num_example_per_class):
    """Randomly select at most N examples per class

    Args:
        x (Tensor): data
        y (Tensor): labels
        class_list (list(int)): list of class to sample from.
        num_example_per_class (int): num of examples to select from EACH class.

    Returns:
        (list): x, y

    """

    # Mapping class to idx
    index_per_class = defaultdict(list)
    for idx in range(len(x)):
        cl = int(y[idx])  # need to cast tensor
        if cl in class_list:
            index_per_class[cl].append(idx)

    # select
    idxs = []
    for class_id in class_list:
        class_idxs = index_per_class[class_id]
        random.shuffle(class_idxs)
        idxs.extend(class_idxs[:num_example_per_class])

    random.shuffle(idxs)
    batch_x = tf.gather(x, idxs)
    batch_y = tf.gather(y, idxs)

    return batch_x, batch_y


class MultiShotMemorySampler(Sequence):

    def __init__(self,
                 x,
                 y,
                 class_per_batch,
                 batch_size=32,
                 batch_per_epoch=1000,
                 augmenter=None):

        self.x = x
        self.y = y
        self.class_per_batch = class_per_batch
        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch
        self.augmenter = augmenter

        # precompute information we need
        # !  can't have less than 2 example per class in a batch otherwise
        # !TL is meanigless
        self.example_per_class = max(batch_size // class_per_batch, 2)
        class_list, _ = tf.unique(y)
        self.class_list = [int(e) for e in class_list]

        # Mapping class to idx
        # In this sampler, contrary to file based one, the pool of examples
        # wont' change so we can optimize by doing this step in the constructor
        self.index_per_class = defaultdict(list)
        for idx in tqdm(range(len(x)), desc='indexing classes'):
            cl = int(y[idx])  # need to cast tensor
            self.index_per_class[cl].append(idx)

    def __len__(self):
        return self.batch_per_epoch

    def on_epoch_end(self):
        pass

    def __getitem__(self, batch_id):
        return self.generate_batch(batch_id)

    def generate_batch(self, batch_id):

        # select class at ramdom
        random.shuffle(self.class_list)
        class_list = self.class_list[:self.class_per_batch]

        # get example for each class
        idxs = []
        for class_id in class_list:
            class_idxs = self.index_per_class[class_id]
            random.shuffle(class_idxs)
            idxs.extend(class_idxs[:self.example_per_class])

        random.shuffle(idxs)
        batch_x = tf.gather(self.x, idxs[:self.batch_size])
        batch_y = tf.gather(self.y, idxs[:self.batch_size])

        if self.augmenter:
            batch_x = self.augmenter(batch_x)
        return batch_x, batch_y


class SingleShotMemorySampler():
    def __init__(self, x, y, batch_size, augmenter):
        pass
