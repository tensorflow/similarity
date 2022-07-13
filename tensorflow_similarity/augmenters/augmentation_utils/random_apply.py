from typing import Callable

import tensorflow as tf

from tensorflow_similarity.types import Tensor


def random_apply(func: Callable[[Tensor], Tensor], p: float, x: Tensor) -> Tensor:
    """Randomly apply function func to x with probability p."""
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32),
        ),
        lambda: func(x),
        lambda: x,
    )
