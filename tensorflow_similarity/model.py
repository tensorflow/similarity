from collections import defaultdict
from tqdm.auto import tqdm
from tabulate import tabulate
from pathlib import Path
import tensorflow as tf
from tensorflow.python.keras.engine import functional
from tensorflow_similarity.indexer import Indexer
from tensorflow_similarity.metrics import EvalMetric, make_metric

from typing import List, Dict, Union
from .types import FloatTensorLike, PandasDataFrame
from .distances import metric_name_canonializer


THRESHOLDS_TARGETS = {
    "very_likely": 0.99,
    "likely": 0.9
}


@tf.keras.utils.register_keras_serializable(package="Similarity")
class SimilarityModel(functional.Functional):
    """Sub-classing Keras.Model to allow access to the forward pass values for
    efficient metric-learning.
    """

    def __init__(self, *args, **kwargs):
        super(SimilarityModel, self).__init__(*args, **kwargs)
        self._index: Indexer = None  # index reference

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y,
                                      y_pred,
                                      regularization_losses=self.losses)

            # FIXME: callback
            # self.distances = cosine_distance(y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def compile(self,
                optimizer='rmsprop',
                distance='auto',
                loss=None,
                metrics=None,
                loss_weights=None,
                table='memory',
                matcher='nmslib_hnsw',
                stat_buffer_size=1000,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        "Configures the model for training"

        # Fetching the distance used from the first loss if auto
        if distance == 'auto':
            if isinstance(loss, list):
                metric_loss = loss[0]
            else:
                metric_loss = loss

            try:
                distance = metric_loss.distance
            except:  # noqa
                msg = "distance='auto' only works if the first loss is a\
                       metric loss"

                raise ValueError(msg)
        else:
            distance = metric_name_canonializer(distance)

        # init index
        self._index: Indexer = Indexer(distance=distance,
                                       table=table,
                                       match_algorithm=matcher,
                                       stat_buffer_size=stat_buffer_size)

        # call underlying keras method
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly,
                        **kwargs)

    def index(self, x, y=None, store_data=True, build=True, verbose=1):
        if verbose:
            print('[Indexing %d points]' % len(x))
            print('|-Computing embeddings')
        embeddings = self.predict(x)
        data = x if store_data else None
        self._index.batch_add(embeddings,
                              y,
                              data,
                              build=build,
                              verbose=verbose)

    def _lookup(self, x, k=5, threads=4):
        print("THIS FUNCTION RETURN BOGUS DISTANCES")
        embeddings = self.predict(x)
        return self._index.batch_lookup(embeddings, k=k)

    def single_lookup(self, x, k=5):
        x = tf.expand_dims(x, axis=0)
        embedding = self.predict(x)[0]
        return self._index.single_lookup(embedding, k=k)

    def index_summary(self):
        self._index.print_stats()

    def calibrate(self,
                  x: List[FloatTensorLike],
                  y: List[int],
                  thresholds_targets: Dict[str, float] = THRESHOLDS_TARGETS,
                  k: int = 1,
                  calibration_metric: Union[str, EvalMetric] = "f1_score",
                  extra_metrics: List[Union[str, EvalMetric]] = ['precision', 'recall'],  # noqa
                  rounding: int = 2,
                  verbose: int = 1):

        # predict
        embeddings = self.predict(x)

        # calibrate
        return self._index.calibrate(embeddings=embeddings,
                                     y=y,
                                     thresholds_targets=thresholds_targets,
                                     k=k,
                                     calibration_metric=calibration_metric,
                                     extra_metrics=extra_metrics,
                                     rounding=rounding,
                                     verbose=verbose)

    def match(self,
              x,
              cutpoint='optimal',
              no_match_label=-1,
              verbose=0):
        """Match a set of examples against the calibrated index

        For the match function to work, the index must be calibrated using
        calibrate().

        Args:
            x (tensor): examples to be matched against the index.

            cutpoint (str, optional): What calibration threshold to use.
            Defaults to 'optimal' which is the optimal F1 threshold computed
            with calibrate().

            no_match_label (int, optional): What label value to assign when
            there is no match. Defaults to -1.

            verbose (int, optional). Defaults to 0.

        Returns:
            list(int): Return which class matches for each supplied example

        Notes:
            This function matches all the cutpoints at once internally as there
            is little performance downside to do so and allows to do the
            evaluation in a single go.

        """
        # basic checks
        if not self._index.is_calibrated():
            raise ValueError('Uncalibrated model: run model.calibration()')

        # get embeddings
        embeddings = self.predict(x)

        # matching
        matches = self._index.match(embeddings,
                                    no_match_label=no_match_label,
                                    verbose=verbose)

        # select which matches to return
        if cutpoint == 'all':  # returns all the cutpoints for eval purpose.
            return matches
        else:  # normal match behavior - returns a specific cut point
            return matches[cutpoint]

    def evaluate_matching(self,
                          x,
                          y,
                          k=1,
                          extra_metrics=['accuracy', 'precision', 'recall'],
                          verbose=1):
        """Evaluate model matching accuracy on a given dataset.

        Args:
            x (tensor): Examples to be matched against the index.
            y (tensor): [description]

            no_match_label (int, optional):  What class value to assign when
            there is no match. Defaults to -1.

            verbose (int, optional): Display results if set to 1 otherwise
            results are returned silently. Defaults to 1.

        Returns:
            dict: evaluation metrics
        """
        # There is some code duplication in this function but that is the best
        # solution to keep the end-user API clean and doing inferences once.

        if not self._index.is_calibrated():
            raise ValueError('Uncalibrated model: run model.calibration()')

        # get embeddings
        if verbose:
            print("|-Computing embeddings")
        embeddings = self.predict(x)

        results = defaultdict(dict)
        cp_metric_name = self.index.calibration_metric.name

        if verbose:
            pb = tqdm(total=len(self.index.cutpoints),
                      desc='Evaluating cutpoints')

        for cp_name, cp_value in self.index.cutpoints.items():
            # create a metric that match at the requested k and threshold
            metric = make_metric(cp_metric_name)
            metric.k = k
            metric.distance_threshold = cp_value
            metrics = [metric] + extra_metrics
            results[cp_name] = self.index.evaluate(embeddings, y, metrics, k)
            if verbose:
                pb.update()
        if verbose:
            pb.close()

        if verbose:
            headers = ['cp', cp_metric_name]  # noqa
            for k in results['optimal'].keys():
                if k not in headers:
                    headers.append(str(k))
            rows = []
            for data in results.values():
                rows.append([data[v] for v in headers])
            print(tabulate(rows, headers=headers))

        return results

    def reset_index(self):
        "Reinitialize the index"
        self._index.reset()

    def init_index(self,
                   distance,
                   table='memory',
                   matcher='nmslib_hnsw',
                   stat_buffer_size=1000):
        "Init the index manually"
        self._index: Indexer = Indexer(distance=distance,
                                       table=table,
                                       match_algorithm=matcher,
                                       stat_buffer_size=stat_buffer_size)

    def load_index(self, filepath):
        index_path = Path(filepath) / "index"
        self._index = Indexer.load(index_path)

    def save_index(self, filepath, compression=True):
        index_path = Path(filepath) / "index"
        self._index.save(index_path, compression=compression)

    def save(self,
             filepath,
             save_index=True,
             compression=True,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None,
             save_traces=True):

        # save trace doesn't exist prior to 2.4 -- asking for it but not
        # using it

        # call underlying keras method to save the mode graph and its weights
        tf.keras.models.save_model(self,
                                   filepath,
                                   overwrite=overwrite,
                                   include_optimizer=include_optimizer,
                                   save_format=save_format,
                                   signatures=signatures,
                                   options=options)
        if self._index and save_index:
            self.save_index(filepath, compression=compression)
        else:
            print('Index not saved as save_index=False')

    def to_data_frame(self, num_items: int = 0) -> PandasDataFrame:
        """Export data as pandas dataframe

        Args:
            num_items (int, optional): Num items to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            pd.DataFrame: a pandas dataframe.
        """
        return self._index.to_data_frame(num_items=num_items)

    # @classmethod
    # def from_config(cls, config):
    #     print('here')
    #     print(config)
    #     del config['name']
    #     return super().from_config(**config)
