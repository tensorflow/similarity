from typing import Callable, Optional, Union

import tensorflow as tf

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.losses import MetricLoss
from tensorflow_similarity.losses.multisim_loss import multisimilarity_loss
from tensorflow_similarity.types import FloatTensor


def _add_memory_variable(tensor):
    """Creates an empty variable with same shape and dtype as `tensor`."""
    shape_no_batch = tensor.shape[1:]
    dtype = tensor.dtype
    init_value = tf.constant([], shape=[0, *shape_no_batch], dtype=dtype)
    var = tf.Variable(
        initial_value=init_value,
        shape=[None, *shape_no_batch],
        dtype=dtype,
        trainable=False,
    )
    return var


class XBM(MetricLoss):
    def __init__(
        self,
        fn: Callable,
        memory_size: int,
        warmup_steps: int = 0,
        reduction: Callable = tf.keras.losses.Reduction.AUTO,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(fn, reduction, name, **kwargs)

        if isinstance(fn, MetricLoss):
            self.distance = fn.distance
            self.fn = fn.fn
            self._fn_kwargs.update(fn._fn_kwargs)
        elif "distance" in kwargs:
            self.distance = kwargs["distance"]

        self._fn_kwargs["remove_diagonal"] = False

        self.memory_size = memory_size
        self.warmup_steps = warmup_steps
        self._y_true_memory = None
        self._y_pred_memory = None
        self._total_steps = tf.Variable(0, dtype=tf.int64, trainable=False)

    def call(self, y_true: FloatTensor, y_pred: FloatTensor) -> FloatTensor:
        with tf.device("/cpu:0"):
            # Build memory from first batch
            if self._y_true_memory is None:
                self._y_true_memory = _add_memory_variable(y_true)
            if self._y_pred_memory is None:
                self._y_pred_memory = _add_memory_variable(y_pred)

        # Enqueue (concat batch to beginning of memory)
        y_true_mem = tf.concat([y_true, self._y_true_memory], axis=0)
        y_pred_mem = tf.concat([y_pred, self._y_pred_memory], axis=0)

        # Dequeue (truncate to memory size limit)
        y_true_mem = y_true_mem[: self.memory_size]
        y_pred_mem = y_pred_mem[: self.memory_size]

        def _xbm_step():
            # Update memory with new values
            # FIXME: WARNING:tensorflow:5 out of the last 5 calls to <function multisimilarity_loss at 0x7fbca3889550> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
            #   Because shape is changing, we get multiple retracings ONLY in EagerMode
            self._y_true_memory.assign(y_true_mem)
            self._y_pred_memory.assign(y_pred_mem)
            return y_true_mem, y_pred_mem

        def _warmup_step():
            # Memory is not updated during warmup steps, so it remains empty.
            # Therefore this corresponds to the original batch
            return y_true_mem, y_pred_mem

        self._total_steps.assign_add(1)
        y_true_mem, y_pred_mem = tf.cond(
            self._total_steps > self.warmup_steps,
            true_fn=_xbm_step,
            false_fn=_warmup_step,
        )

        loss: FloatTensor = self.fn(y_true, y_pred, y_true_mem, y_pred_mem, **self._fn_kwargs)
        return loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class XBMMultiSimilarityLoss(XBM):
    """Computes the multi similarity loss in an online fashion.


    `y_true` must be  a 1-D integer `Tensor` of shape (batch_size,).
    It's values represent the classes associated with the examples as
    **integer  values**.

    `y_pred` must be 2-D float `Tensor`  of L2 normalized embedding vectors.
    you can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
    last layer of your model to ensure your model output is properly
    normalized.
    """

    def __init__(
        self,
        memory_size: int,
        distance: Union[Distance, str] = "cosine",
        alpha: float = 1.0,
        beta: float = 20,
        epsilon: float = 0.2,
        lmda: float = 0.5,
        warmup_steps: int = 0,
        name: str = "XBMMultiSimilarityLoss",
        **kwargs,
    ):
        """Initializes the Multi Similarity Loss

        Args:
            distance: Which distance function to use to compute the pairwise
            distances between embeddings. Defaults to 'cosine'.

            alpha: The exponential weight for the positive pairs. Increasing
            alpha makes the logsumexp softmax closer to the max positive pair
            distance, while decreasing it makes it closer to
            max(P) + log(batch_size).

            beta: The exponential weight for the negative pairs. Increasing
            beta makes the logsumexp softmax closer to the max negative pair
            distance, while decreasing it makes the softmax closer to
            max(N) + log(batch_size).

            epsilon: Used to remove easy positive and negative pairs. We only
            keep positives that we greater than the (smallest negative pair -
            epsilon) and we only keep negatives that are less than the
            (largest positive pair + epsilon).

            lmda: Used to weight the distance. Below this distance, negatives
            are up weighted and positives are down weighted. Similarly, above
            this distance negatives are down weighted and positive are up
            weighted.

            name: Loss name. Defaults to MultiSimilarityLoss.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance

        super().__init__(
            multisimilarity_loss,
            name=name,
            distance=distance,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            lmda=lmda,
            memory_size=memory_size,
            warmup_steps=warmup_steps,
            remove_diagonal=False,
            **kwargs
        )
