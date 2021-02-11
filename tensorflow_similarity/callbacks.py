from tensorflow.keras.callbacks import Callback
from tensorflow_similarity.types import TensorLike
from typing import List, DefaultDict
from collections import defaultdict
import tensorflow as tf


class EvalCallback(Callback):

    def __init__(self,
                 queries: TensorLike,
                 query_labels: List[int],
                 targets: TensorLike,
                 target_labels: List[int],
                 distance: str = 'cosine',
                 tb_logdir: str = None,
                 k: int = 1):
        super().__init__()
        self.queries = queries
        self.queries_labels = query_labels
        self.targets = targets
        self.targets_labels = target_labels
        self.distance = distance
        self.k = k

        if tb_logdir:
            tb_logdir = tb_logdir + '/match_rate/'
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)
            print('TensorBoard logging enable in %s' % tb_logdir)
        else:
            self.tb_writer = None

    def on_epoch_end(self, epoch: int, logs: dict):
        # reset the index
        self.model.reset_index()

        # rebuild the index
        self.model.index(self.targets, self.targets_labels, verbose=0)

        # query the index and count the match
        matches: DefaultDict[str, int] = defaultdict(int)
        for qidx, query in enumerate(self.queries):
            query_label = self.queries_labels[qidx]
            nn = self.model.single_lookup(query, k=self.k)
            matched = 0  # recall matches are cummulative
            for n in nn:
                if n['label'] == query_label:
                    matched = 1
                matches[n['rank']] += matched

        # stats and write to the logs
        self.match_rates = []  # we use an array for stable display
        for idx in range(len(matches)):
            rank = idx + 1
            match_rate = matches[rank] / len(self.queries)
            metric_name = "match_rate_@%s" % (rank)
            logs[metric_name] = match_rate
            self.match_rates.append([metric_name, match_rate])

        # for now just display till logs are fixed
        metrics = ['%s:%0.4f' % (m[0], m[1]) for m in self.match_rates]
        print(' - '.join(metrics))

        # Tensorboard
        if self.tb_writer:
            with self.tb_writer.as_default():
                for m in self.match_rates:
                    tf.summary.scalar(m[0], m[1], step=epoch)
