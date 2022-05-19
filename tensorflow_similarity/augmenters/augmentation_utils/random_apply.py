import tensorflow as tf

from typing import Callable


def random_apply(
    func: Callable[[tf.Tensor], tf.Tensor], p: float, x: tf.Tensor
) -> tf.Tensor:
    """Randomly apply function func to x with probability p."""
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32),
        ),
        lambda: func(x),
        lambda: x,
    )
