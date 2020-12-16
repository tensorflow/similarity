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