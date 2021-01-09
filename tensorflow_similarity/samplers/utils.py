from collections import defaultdict
import random
from typing import Sequence, Tuple

import tensorflow as tf
from tensorflow.types import experimental as tf_types


def select_examples(x: tf_types.TensorLike,
                    y: tf_types.TensorLike,
                    class_list: Sequence[int],
                    num_example_per_class: int
                    ) -> Tuple[tf_types.TensorLike, tf_types.TensorLike]:
    """Randomly select at most N examples per class

    Args:
        x: A 2-D Tensor containing the data.
        y: A 1-D Tensor containing the labels.
        class_list: list of class to sample from.
        num_example_per_class: num of examples to select from EACH class.

    Returns:
        A Tuple containing the subset of x and y.
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
        idxs.extend(random.choices(class_idxs, k=num_example_per_class))

    random.shuffle(idxs)
    batch_x = tf.gather(x, idxs)
    batch_y = tf.gather(y, idxs)

    return batch_x, batch_y
