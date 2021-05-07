from typing import List, Union
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tensorflow_similarity.types import Tensor
from tensorflow_similarity.evaluators import MemoryEvaluator
from tensorflow_similarity.metrics import EvalMetric, make_metric
from .types import FloatTensor, IntTensor


class EvalCallback(Callback):
    """Epoch end evaluation callback that build a test index and evaluate
    model performance on it.

    This evaluation only run at epoch_end as it is computationally very
    expensive.

    """

    def __init__(self,
                 queries: Tensor,
                 query_labels: List[int],
                 targets: Tensor,
                 target_labels: List[int],
                 distance: str = 'cosine',
                 metrics: List[Union[str, EvalMetric]] = ['accuracy', 'mean_rank'],  # noqa
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
        self.queries_labels = query_labels
        self.targets = targets
        self.targets_labels = target_labels
        self.distance = distance
        self.k = k
        self.index_size = len(target_labels)
        self.evaluator = MemoryEvaluator()
        # typing requires this weird formulation of creating a new list
        self.metrics: List[Union[str, EvalMetric]] = [make_metric(m) for m in metrics] # noqa

        if tb_logdir:
            tb_logdir = str(Path(tb_logdir) / 'index/')
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)
            print('TensorBoard logging enable in %s' % tb_logdir)
        else:
            self.tb_writer = None

    def on_epoch_end(self, epoch: int, logs: dict):
        # reset the index
        self.model.reset_index()

        # rebuild the index
        self.model.index(self.targets, self.targets_labels, verbose=0)

        lookups = self.model.lookup(self.queries)
        results = self.evaluator.evaluate(self.index_size, self.metrics,
                                          self.queries_labels, lookups)

        # for now just display till tensorflow logs allows to write
        mstr = ['%s:%0.4f' % (k, v) for k, v in results.items()]
        print(' - '.join(mstr))

        # Tensorboard if configured
        if self.tb_writer:
            with self.tb_writer.as_default():
                for k, v in results.items():
                    tf.summary.scalar(k, v, step=epoch)


class SplitValidationLoss(Callback):
    """A split validation callback.

    This callback will split the validation data into two sets.

        1) The set of classes seen during training.
        2) The set of classes not seen during training.

    The callback will then compute a separate validation for each split.

    Attributes:
        history:
        x_known:
        y_known:
        x_unkown:
        y_unkown:
    """
    def __init__(self,
                 x: FloatTensor,
                 y: IntTensor,
                 known_classes: np.ndarray):
        """Creates the validation callbacks.

        Args:
            x: Validation data.
            y: Validation labels.
            known_classes: The set of classes seen during training.
        """
        super().__init__()
        self.history = {}

        # Create separate validation sets for the known and unknon classes
        broadcast_classes = known_classes[np.newaxis, :]
        broadcast_labels = y[:, np.newaxis]
        known_mask = np.any(broadcast_classes == broadcast_labels, axis=1)
        known_idxs = np.argwhere(known_mask)
        unknown_idxs = np.argwhere(~known_mask)

        self.x_known = tf.gather(x, indices=known_idxs.reshape(-1))
        self.y_known = y[known_idxs]
        self.x_unknown = tf.gather(x, indices=unknown_idxs.reshape(-1))
        self.y_unkown = y[unknown_idxs]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        known_eval = self.model.evaluate(self.x_known, self.y_known, verbose=0)
        unknown_eval = (
                self.model.evaluate(self.x_unknown, self.y_unkown, verbose=0))
        print(f'val_los - known_classes: {known_eval:.4f} - '
              f'unkown_classes: {unknown_eval:.4f}')
        logs['known_val_loss'] = known_eval
        logs['unknown_val_loss'] = unknown_eval
