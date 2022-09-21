# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ArcFace losses  base class.

ArcFace: Additive Angular Margin Loss for Deep Face
Recognition. [online] arXiv.org. Available at:
<https://arxiv.org/abs/1801.07698v3>.

"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import tensorflow as tf

from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import FloatTensor, IntTensor
from tensorflow_similarity.utils import is_tensor_or_variable

from .metric_loss import MetricLoss
from .utils import logsumexp


@tf.keras.utils.register_keras_serializable(package="Similarity")
class ArcFaceLoss(tf.keras.losses.Loss):
    """Implement of ArcFace: Additive Angular Margin Loss:
    Step 1: Create a trainable kernel matrix with the shape of [embedding_size, num_classes].
    Step 2: Normalize the kernel and prediction vectors.
    Step 3: Calculate the cosine similarity between the normalized prediction vector and the kernel.
    Step 4: Create a one-hot vector include the margin value for the ground truth class.
    Step 5: Add margin_hot to the cosine similarity and multiply it by scale.
    Step 6: Calculate the cross-entropy loss.

    ArcFace: Additive Angular Margin Loss for Deep Face
             Recognition. [online] arXiv.org. Available at:
             <https://arxiv.org/abs/1801.07698v3>.

    Standalone usage:
        >>> loss_fn = tfsim.losses.ArcFaceLoss(num_classes=2, embedding_size=3)
        >>> labels = tf.Variable([1, 0])
        >>> embeddings = tf.Variable([[0.2, 0.3, 0.1], [0.4, 0.5, 0.5]])
        >>> loss = loss_fn(labels, embeddings)
    Args:
        num_classes: Number of classes.
        embedding_size: The size of the embedding vectors.
        margin: The margin value.
        scale: s in the paper, feature scale
        name: Optional name for the operation.
        reduction: Type of loss reduction to apply to the loss.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin: float = 0.50,  # margin in radians
        scale: float = 64.0,  # feature scale
        name: Optional[str] = None,
        reduction: Callable = tf.keras.losses.Reduction.AUTO,
        **kwargs
    ):

        super().__init__(reduction=reduction, name=name, **kwargs)

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        self.name = name
        self.kernel = tf.Variable(tf.random.normal([embedding_size, num_classes]))

    def call(self, y_true: FloatTensor, y_pred: FloatTensor) -> FloatTensor:

        y_pred_norm = tf.math.l2_normalize(y_pred, axis=1)
        kernel_norm = tf.math.l2_normalize(self.kernel, axis=0)

        cos_theta = tf.matmul(y_pred_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)

        m_hot = tf.one_hot(y_true, self.num_classes, on_value=self.margin, axis=1)
        m_hot = tf.reshape(m_hot, [-1, self.num_classes])

        cos_theta = tf.acos(cos_theta)
        cos_theta += m_hot
        cos_theta = tf.math.cos(cos_theta)
        cos_theta = tf.math.multiply(cos_theta, self.scale)

        cce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=self.reduction
        )
        loss: FloatTensor = cce(y_true, cos_theta)

        return loss

    def get_config(self) -> Dict[str, Any]:
        """Contains the loss configuration.
        Returns:
            A Python dict containing the configuration of the loss.
        """
        config = {
            "num_classes": self.num_classes,
            "embedding_size": self.embedding_size,
            "margin": self.margin,
            "scale": self.scale,
            "name": self.name,
        }
        base_config = super().get_config()
        return {**base_config, **config}
