from typing import List, Sequence, Union
from pathlib import Path
import math

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from .classification_metrics import ClassificationMetric
from .classification_metrics import make_classification_metric # noqa
from .evaluators import MemoryEvaluator
from .types import Tensor, FloatTensor, IntTensor
from .utils import unpack_lookup_distances, unpack_lookup_labels


class EvalCallback(Callback):
    """Epoch end evaluation callback that build a test index and evaluate
    model performance on it.

    This evaluation only run at epoch_end as it is computationally very
    expensive.

    """

    def __init__(self,
                 queries: Tensor,
                 query_labels: Sequence[int],
                 targets: Tensor,
                 target_labels: Sequence[int],
                 distance: str = 'cosine',
                 metrics: Sequence[Union[str, ClassificationMetric]] = ['accuracy', 'f1score'],  # noqa
                 tb_logdir: str = None,
                 k: int = 1):
        """Evaluate model matching quality against a validation dataset at
        epoch end.

        Args:
            queries: Test examples that will be tested against the built index.

            query_labels: Queries nearest neighboors expected labels.

            targets: Examples that are indexed.

            target_labels: Target examples labels.

            distance: Distance function used to compute pairwise distance
            between examples embeddings.

            metrics: List of [EvalMetrics](eval_metrics.md) to be computed
            during the evaluation. Defaults to ['accuracy', 'mean_rank'].
            embedding to evaluate.

            tb_logdir: Where to write TensorBoard logs. Defaults to None.

            k: How many neigboors to retrive for evaluation. Defaults to 1.

        """
        super().__init__()
        self.queries = queries
        self.query_labels: IntTensor = tf.cast(
                tf.convert_to_tensor(query_labels),
                dtype='int32'
        )
        self.targets = targets
        self.target_labels = target_labels
        self.distance = distance
        self.k = k
        self.evaluator = MemoryEvaluator()
        # typing requires this weird formulation of creating a new list
        self.metrics: List[ClassificationMetric] = (
                [make_classification_metric(m) for m in metrics])

        if tb_logdir:
            tb_logdir = str(Path(tb_logdir) / 'index/')
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)
            print('TensorBoard logging enable in %s' % tb_logdir)
        else:
            self.tb_writer = None

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if logs is None:
            logs = {}

        # reset the index
        self.model.reset_index()

        # rebuild the index
        self.model.index(self.targets, self.target_labels, verbose=0)

        lookups = self.model.lookup(self.queries, verbose=0)
        lookup_distances = unpack_lookup_distances(lookups)
        lookup_labels = unpack_lookup_labels(lookups)

        results = self.evaluator.evaluate_classification(
            query_labels=self.query_labels,
            lookup_labels=lookup_labels,
            lookup_distances=lookup_distances,
            distance_thresholds=tf.constant([math.inf]),
            metrics=self.metrics,
            matcher='match_nearest',
            verbose=0)

        mstr = []
        for metric_name, vals in results.items():
            float_val = vals[0]
            logs[metric_name] = float_val
            mstr.append(f'{metric_name}:{float_val:0.4f}')
            if self.tb_writer:
                with self.tb_writer.as_default():
                    tf.summary.scalar(metric_name, float_val, step=epoch)

        print(' - '.join(mstr))


class SplitValidationLoss(Callback):
    """A split validation callback.

    This callback will split the validation data into two sets.

        1) The set of classes seen during training.
        2) The set of classes not seen during training.

    The callback will then compute a separate validation for each split.

    This is useful for separately tracking the validation loss on the seen and
    unseen classes and may provide insight into how well the embedding will
    generalize to new classes.

    Attributes:
        x_known: The set of examples from the known classes.
        y_known: The labels associated with the known examples.
        x_unknown: The set of examples from the unknown classes.
        y_unknown: The labels associated with the unknown examples.
    """
    def __init__(self,
                 x: FloatTensor,
                 y: IntTensor,
                 known_classes: IntTensor):
        """Creates the validation callbacks.

        Args:
            x: Validation data.
            y: Validation labels.
            known_classes: The set of classes seen during training.
        """
        super().__init__()

        # Create separate validation sets for the known and unknown classes
        y = tf.cast(y, dtype=tf.int32)
        known_classes = tf.cast(known_classes, dtype=tf.int32)
        known_classes = tf.squeeze(known_classes)

        # Use broadcasting to do a y X known_classes equality check. By adding
        # a dim to the start of known_classes and a dim to the end of y, this
        # essentially checks `for ck in known_classes: for cy in y: ck == cy`.
        # We then reduce_any to find all rows in y that match at least one
        # class in known_classes.
        # See https://numpy.org/doc/stable/user/basics.broadcasting.html
        broadcast_classes = tf.expand_dims(known_classes, axis=0)
        broadcast_labels = tf.expand_dims(y, axis=-1)
        known_mask = tf.math.reduce_any(
                broadcast_classes == broadcast_labels, axis=1)
        known_idxs = tf.squeeze(tf.where(known_mask))
        unknown_idxs = tf.squeeze(tf.where(~known_mask))

        with tf.device("/cpu:0"):
            self.x_known = tf.gather(x, indices=known_idxs)
            self.y_known = tf.gather(y, indices=known_idxs)

            self.x_unknown = tf.gather(x, indices=unknown_idxs)
            self.y_unknown = tf.gather(y, indices=unknown_idxs)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        _ = epoch
        if logs is None:
            logs = {}
        known_eval = self.model.evaluate_classification(
                self.x_known,
                self.y_known,
                verbose=0)
        unknown_eval = self.model.evaluate_classification(
                self.x_unknown,
                self.y_unknown,
                verbose=0)
        print(f'val_los - known_classes: {known_eval:.4f} - '
              f'unknown_classes: {unknown_eval:.4f}')
        logs['known_val_loss'] = known_eval
        logs['unknown_val_loss'] = unknown_eval
