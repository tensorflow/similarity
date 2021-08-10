from tqdm.auto import tqdm
from collections import defaultdict
import random
from typing import Sequence, Tuple

import tensorflow as tf
from tensorflow_similarity.types import IntTensor, FloatTensor


def select_examples(x: FloatTensor,
                    y: IntTensor,
                    class_list: Sequence[int] = None,
                    num_examples_per_class: int = None,
                    ) -> Tuple[FloatTensor, IntTensor]:
    """Randomly select at most N examples per class

    Args:
        x: A 2-D Tensor containing the data.

        y: A 1-D Tensor containing the labels.

        class_list: Filter the list of examples to only keep thoses those who
        belong to the supplied class list. In no class is supplied, keep
        examples for all the classes. Default to None - keep all the examples.

        num_examples_per_class: Restrict the number of examples for EACH
        class to num_examples_per_class if set. If not set, all the available
        examples are selected. Defaults to None - no selection.

    Returns:
        A Tuple containing the subset of x and y.
    """

    # cast class_list if it exist to avoid slowness
    if class_list is not None:
        class_list_int = [int(c) for c in class_list]
    else:
        class_list_int = list(set([int(e) for e in y]))

    # Mapping class to idx
    index_per_class = defaultdict(list)
    cls = [int(c) for c in y]

    for idx in tqdm(range(len(x)), desc="filtering classes"):
        cl = cls[idx]  # need to cast tensor

        # if user provided a class_list, check it's part of it.
        if class_list is not None and cl in class_list_int:
            index_per_class[cl].append(idx)
        else:
            # just add all class examples
            index_per_class[cl].append(idx)

    # restrict numbers of samples
    idxs = []
    for class_id in tqdm(class_list_int, desc="selecting examples"):
        class_idxs = index_per_class[class_id]

        # restrict num examples?
        if num_examples_per_class:
            idxs.extend(random.choices(class_idxs, k=num_examples_per_class))
        else:
            idxs.extend(class_idxs)

    random.shuffle(idxs)
    batch_x = tf.gather(x, idxs)
    batch_y = tf.gather(y, idxs)

    return batch_x, batch_y
