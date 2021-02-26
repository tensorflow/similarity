from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from pathlib import Path

from typing import List, Union
from tensorflow_similarity.types import TensorLike
from tensorflow_similarity.evaluators import MemoryEvaluator
from tensorflow_similarity.metrics import EvalMetric, make_metric


class EvalCallback(Callback):

    def __init__(self,
                 queries: TensorLike,
                 query_labels: List[int],
                 targets: TensorLike,
                 target_labels: List[int],
                 distance: str = 'cosine',
                 metrics: List[Union[str, EvalMetric]] = ['accuracy', 'mean_rank'],  # noqa
                 tb_logdir: str = None,
                 eval_head: int = None,
                 k: int = 1):
        """Evaluate model matching quality against a validation dataset at
        epoch end.

        Args:
            queries (TensorLike): [description]
            query_labels (List[int]): [description]
            targets (TensorLike): [description]
            target_labels (List[int]): [description]
            distance (str, optional): [description]. Defaults to 'cosine'.
            metrics (List[Union[str, EvalMetric]], optional): [description]. Defaults to ['accuracy', 'mean_rank'].
            embedding to evaluate. Defaults to None which is for model
            with a single head.
            k (int, optional): [description]. Defaults to 1.
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

            tb_logdir = str(Path(tb_logdir) / 'match_rate/')
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)
            print('TensorBoard logging enable in %s' % tb_logdir)
        else:
            self.tb_writer = None

    def on_epoch_end(self, epoch: int, logs: dict):
        # reset the index
        self.model.reset_index()

        # rebuild the index
        self.model.index(self.targets, self.targets_labels, verbose=0)

        lookups = []
        for idx in range(len(self.queries_labels)):
            # FIXME batch lookup when ready
            nn = self.model.single_lookup(self.queries[idx], k=1)
            lookups.append(nn)
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
