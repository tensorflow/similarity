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
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from tqdm.auto import tqdm

from tensorflow_similarity.types import FloatTensor, IntTensor


def select_examples(
    x: FloatTensor,
    y: IntTensor,
    class_list: Sequence[int] | None = None,
    num_examples_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
        class_list_int = set([int(c) for c in class_list])
    else:
        class_list_int = set([int(e) for e in y])

    # Mapping class to idx
    index_per_class = defaultdict(list)
    cls = [int(c) for c in y]

    for idx in tqdm(range(len(x)), desc="filtering examples"):
        cl = cls[idx]  # need to cast tensor

        # if user provided a class_list, check it's part of it.
        if class_list is not None and cl in class_list_int:
            index_per_class[cl].append(idx)
        else:
            # just add all class examples
            index_per_class[cl].append(idx)

    # restrict numbers of samples
    idxs = []
    for class_id in tqdm(class_list_int, desc="selecting classes"):
        class_idxs = index_per_class[class_id]

        # restrict num examples?
        if num_examples_per_class:
            idxs.extend(random.choices(class_idxs, k=num_examples_per_class))
        else:
            idxs.extend(class_idxs)

    random.shuffle(idxs)

    batch_x = []
    batch_y = []
    for idx in tqdm(idxs, desc="gather examples"):
        batch_x.append(x[idx])
        batch_y.append(y[idx])

    return np.array(batch_x), np.array(batch_y)
