from collections import defaultdict
from copy import copy
from pathlib import Path
from tensorflow_similarity.evaluators.evaluator import Evaluator
from typing import Dict, List, Union, DefaultDict, Optional
from tabulate import tabulate
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import Loss

from tensorflow_similarity.indexer import Indexer
from tensorflow_similarity.tables import Table
from tensorflow_similarity.matchers import Matcher
from tensorflow_similarity.metrics import EvalMetric, make_metric
from tensorflow_similarity.distances import Distance
from tensorflow_similarity.distances import distance_canonicalizer
from tensorflow_similarity.distance_metrics import DistanceMetric
from tensorflow_similarity.losses import MetricLoss
from tensorflow_similarity.types import FloatTensor, Lookup, IntTensor, Tensor
from tensorflow_similarity.types import PandasDataFrame


@tf.keras.utils.register_keras_serializable(package="Similarity")
class SimilarityModel(tf.keras.Model):
    """Specialized Keras.Model which implement the core features needed for
    metric learning. In particular, `SimilarityModel()` supports indexing,
    searching and saving the embeddings predicted by the network.

    All Similarity models classes derive from this class to benefits from those
    core features.
    """

    # @property
    # def _index(self):
    #     if not hasattr(self, '_index'):
    #         ValueError("Index doesn't exist: index data before quering it")
    #     return self._index

    # @index.setter
    # def _index(self, index):
    #     self._index: Indexer = index

    def compile(self,
                optimizer: Union[Optimizer, str, Dict, List] = 'rmsprop',  # noqa
                distance: Union[Distance, str] = 'auto',
                loss: Union[Loss, MetricLoss, str, Dict, List] = None,
                metrics: Union[Metric, DistanceMetric, str, Dict, List] = None,
                embedding_output: int = None,
                table: Union[Table, str] = 'memory',
                matcher: Union[Matcher, str] = 'nmslib',
                evaluator: Union[Evaluator, str] = 'memory',
                stat_buffer_size: int = 1000,
                loss_weights: List = None,
                weighted_metrics: List = None,
                run_eagerly: bool = False,
                steps_per_execution: int = 1,
                **kwargs):
        """Configures the model for training.

        Args:

            optimizer: String (name of optimizer) or optimizer instance. See
            [tf.keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers).

            loss: String (name of objective function), objective function,
            any `tensorflow_similarity.loss.*` instance or a
            `tf.keras.losses.Loss` instance. See the [Losses
            documentation](../losses.md) for a list of metric learning
            specifics loss offered by TensorFlow Similairy and
            [tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
            for the losses available directly in TensorFlow.


            metrics: List of metrics to be evaluated by the model during
            training and testing. Each of those can be a string,
            a function or a [tensorflow_similairty.metrics.*](../metrics.md)
            instance. Note that the metrics used for some type of
            metric-learning such as distance learning (e.g via triplet loss)
            have a different prototype than the metrics used in
            standard models and you can't use the `tf.keras.metrics` for those
            type of learning.

            Additionally many distance metrics are computed based of the
            [Indexer()](../indexer.md) performance. E.g Matching Top 1
            accuracy. For technical and performance reasons, indexing data at
            each training batch to compute those is impractical so
            those metrics are computed at epoch end via
            the [EvalCallback](../callbacks.md)

            See [Evaluation Metrics](../eval_metrics.md) for a list of
            available metrics.

            For multi-output models you can specify different metrics for
            different outputs by passing a dictionary, such as
            `metrics={'similarity': 'min_neg_gap', 'other': ['accuracy', 'mse']}`.
            You can also pass a list (len = len(outputs)) of lists of metrics
            such as `metrics=[['min_neg_gap'], ['accuracy', 'mse']]` or
            `metrics=['min_neg_gap', ['accuracy', 'mse']]`. For outputs which
            are not related to metrics learning, you can use any of the
            standard `tf.keras.metrics`.


            loss_weights: Optional list or dictionary specifying scalar
            coefficients (Python floats) to weight the loss contributions of
            different model outputs. The loss value that will be minimized
            by the model will then be the *weighted sum* of all individual
            losses, weighted by the `loss_weights` coefficients.
            If a list, it is expected to have a 1:1 mapping to the model's
            outputs. If a dict, it is expected to map output names (strings)
            to scalar coefficients.

            weighted_metrics: List of metrics to be evaluated and weighted by
            sample_weight or class_weight during training and testing.


            run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s
            logic will not be wrapped in a `tf.function`. Recommended to leave
            this as `None` unless your `Model` cannot be run inside a
            `tf.function`.

            steps_per_execution: Int. Defaults to 1. The number of batches to
            run during each `tf.function` call. Running multiple batches
            inside a single `tf.function` call can greatly improve performance
            on TPUs or small models with a large Python overhead.
            At most, one full epoch will be run each execution. If a number
            larger than the size of the epoch is passed,  the execution will be
            truncated to the size of the epoch.
            Note that if `steps_per_execution` is set to `N`,
            `Callback.on_batch_begin` and `Callback.on_batch_end` methods will
            only be called every `N` batches (i.e. before/after each
            `tf.function` execution).

    Raises:
        ValueError: In case of invalid arguments for
            `optimizer`, `loss` or `metrics`.
    """

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
            print("Distance metric automatically set to", distance,
                  "use the distance arg to override.")
        else:
            distance = distance_canonicalizer(distance)

        # check if we we need to set the embedding head
        num_outputs = len(self.output_names)
        if embedding_output and embedding_output > num_outputs:
            raise Exception("Embedding_output value exceed number of model "
                            "outputs")

        if not embedding_output and num_outputs > 1:
            print("Embedding output set to be model output 0",
                  "Use the embedding_output arg to override this")
            embedding_output = 0

        # fetch embedding size as some ANN libs requires it for init
        if num_outputs > 1:
            self.embedding_size = self.outputs[embedding_output].shape[1]
        else:
            self.embedding_size = self.outputs[0].shape[1]

        # init index
        self._index = Indexer(distance=distance,
                              embedding_size=self.embedding_size,
                              embedding_output=embedding_output,
                              table=table,
                              matcher=matcher,
                              evaluator=evaluator,
                              stat_buffer_size=stat_buffer_size)

        # call underlying keras method
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly,
                        steps_per_execution=steps_per_execution,
                        **kwargs)

    def index(self,
              x: Tensor,
              y: IntTensor = None,
              data: Optional[Tensor] = None,
              build: bool = True,
              verbose: int = 1):
        """Index data.

        Args:
            x: Samples to index.

            y: class ids associated with the data if any. Defaults to None.
            store_data: store the data associated with the samples in Table.
            Defaults to True.

            build: Rebuild the index after indexing. This is needed to make the
            new samples searchable. Set it to false to save processing time
            when calling indexing repeatidly without the need to search between
            the indexing requests. Defaults to True.

            verbose: Output indexing progress info. Defaults to 1.
        """

        if not self._index:
            raise Exception('You need to compile the model with a valid distance to be able to use the indexing')
        if verbose:
            print('[Indexing %d points]' % len(x))
            print('|-Computing embeddings')
        predictions = self.predict(x)
        self._index.batch_add(predictions,
                              y,
                              data,
                              build=build,
                              verbose=verbose)

    def lookup(self,
               x: Tensor,
               k: int = 5,
               verbose: int = 1) -> List[List[Lookup]]:
        """Find the k closest matches in the index for a set of samples.

        Args:
            x: Samples to match.

            k: Number of nearest neighboors to lookup. Defaults to 5.

            verbose: display progress. Default to 1.

        Returns
            list of list of k nearest neighboors:
            List[List[Lookup]]
        """
        predictions = self.predict(x)
        return self._index.batch_lookup(predictions,
                                        k=k,
                                        verbose=verbose)

    def single_lookup(self,
                      x: Tensor,
                      k: int = 5) -> List[Lookup]:
        """Find the k closest matches in the index for a given sample.

        Args:
            x: Sample to match.

            k: Number of nearest neighboors to lookup. Defaults to 5.

        Returns
            list of the k nearest neigboors info:
            List[Lookup]
        """
        x = tf.expand_dims(x, axis=0)
        prediction = self.predict(x)
        return self._index.single_lookup(prediction, k=k)

    def index_summary(self):
        "Display index info summary."
        self._index.print_stats()

    def calibrate(self,
                  x: FloatTensor,
                  y: IntTensor,
                  thresholds_targets: Dict[str, float] = {},
                  k: int = 1,
                  calibration_metric: Union[str, EvalMetric] = "f1_score",
                  extra_metrics: List[Union[str, EvalMetric]] = ['accuracy', 'recall'],  # noqa
                  rounding: int = 2,
                  verbose: int = 1):
        """Calibrate model thresholds using a test dataset.
            FIXME: more detailed explaination.

            Args:

                x: examples to use for the calibration.

                y: Expected labels for the nearest neighboors.

                thresholds_targets: Dict of performance targets to
                (if possible) meet with respect to the `calibration_metric`.

                calibration_metric: [Metric()](metrics/overview.md) used to
                evaluate the performance of the index.

                k: How many neighboors to use during the calibration.
                Defaults to 1.

                extra_metrics: List of additional [Metric()](
                metrics/overview.md) to compute and report.

                rounding: Metric rounding. Default to 2 digits.

                verbose: Be verbose and display calibration results.
                Defaults to 1.

            Returns:
                Calibration results: `{"cutpoints": {}, "thresholds": {}}`
        """

        # predict
        predictions = self.predict(x)

        # calibrate
        return self._index.calibrate(predictions=predictions,
                                     y=y,
                                     thresholds_targets=thresholds_targets,
                                     k=k,
                                     calibration_metric=calibration_metric,
                                     extra_metrics=extra_metrics,
                                     rounding=rounding,
                                     verbose=verbose)

    def match(self,
              x: FloatTensor,
              cutpoint='optimal',
              no_match_label=-1,
              verbose=0):
        """Match a set of examples against the calibrated index

        For the match function to work, the index must be calibrated using
        calibrate().

        Args:
            x: Batch of examples to be matched against the index.

            cutpoint: Which calibration threshold to use.
            Defaults to 'optimal' which is the optimal F1 threshold computed
            using calibrate().

            no_match_label: Which label value to assign when there is no
            match. Defaults to -1.

            verbose. Be verbose. Defaults to 0.

        Returns:
            List of class ids that matches for each supplied example

        Notes:
            This function matches all the cutpoints at once internally as there
            is little performance downside to do so and allows to do the
            evaluation in a single go.

        """
        # basic checks
        if not self._index.is_calibrated:
            raise ValueError('Uncalibrated model: run model.calibration()')

        # get predictions
        predictions = self.predict(x)

        # matching
        matches = self._index.match(predictions,
                                    no_match_label=no_match_label,
                                    verbose=verbose)

        # select which matches to return
        if cutpoint == 'all':  # returns all the cutpoints for eval purpose.
            return matches
        else:  # normal match behavior - returns a specific cut point
            return matches[cutpoint]

    def evaluate_matching(self,
                          x: Tensor,
                          y: IntTensor,
                          k: int = 1,
                          extra_metrics: List[Union[EvalMetric, str]] = ['accuracy', 'recall'],  # noqa
                          verbose: int = 1):
        """Evaluate model matching accuracy on a given evaluation dataset.

        Args:
            x: Examples to be matched against the index.

            y: Label associated with the examples supplied.

            k: How many neigboors to use to perform the evaluation.
            Defaults to 1.

            extra_metrics: Additional (distance_metrics.mde)[distance metrics]
            to be computed during the evaluation. Defaut to accuracy and
            recall.

            verbose (int, optional): Display results if set to 1 otherwise
            results are returned silently. Defaults to 1.

        Returns:
            Dictionnary of (distance_metrics.md)[evaluation metrics]
        """
        # There is some code duplication in this function but that is the best
        # solution to keep the end-user API clean and doing inferences once.

        if not self._index.is_calibrated:
            raise ValueError('Uncalibrated model: run model.calibration()')

        # get embeddings
        if verbose:
            print("|-Computing embeddings")
        predictions = self.predict(x)

        results: DefaultDict['str', Dict[str, Union[int, float, str]]] = defaultdict(dict)  # noqa
        cal_metric = self._index.get_calibration_metric()

        if verbose:
            pb = tqdm(total=len(self._index.cutpoints),
                      desc='Evaluating cutpoints')

        res: Dict[str, Union[str, int, float]] = {}
        for cp_name, cp_data in self._index.cutpoints.items():
            # create a metric that match at the requested k and threshold
            metric = make_metric(cal_metric.name)
            metric.k = k
            metric.distance_threshold = float(cp_data['distance'])
            metrics = copy(extra_metrics)
            metrics.append(metric)
            res.update(self._index.evaluate(predictions, y, metrics, k))
            res['distance'] = cp_data['distance']
            res['name'] = cp_name
            results[cp_name] = res
            if verbose:
                pb.update()
        if verbose:
            pb.close()

        if verbose:
            headers = ['name', cal_metric.name]
            for i in results['optimal'].keys():
                if i not in headers:
                    headers.append(str(i))
            rows = []
            for data in results.values():
                rows.append([data[v] for v in headers])
            print('\n [Summary]\n')
            print(tabulate(rows, headers=headers))

        return results

    def reset_index(self):
        "Reinitialize the index"
        self._index.reset()

    def index_size(self) -> int:
        "Return the index size"
        return self._index.size()

    def load_index(self, filepath: str):
        """Load Index data from a checkpoint and initialize underlying
        structure with the reloaded data.

        Args:
            path: Directory where the checkpoint is located.
            verbose: Be verbose. Defaults to 1.
        """

        index_path = Path(filepath) / "index"
        self._index = Indexer.load(index_path)

    def save_index(self, filepath, compression=True):
        """Save the index to disk

        Args:
            path: directory where to save the index
            compression: Store index data compressed. Defaults to True.
        """
        index_path = Path(filepath) / "index"
        self._index.save(index_path, compression=compression)

    def save(self,
             filepath: str,
             save_index: bool = True,
             compression: bool = True,
             overwrite: bool = True,
             include_optimizer: bool = True,
             signatures=None,
             options=None,
             save_traces: bool = True):
        """Save the model and the index.

        Args:
            filepath: where to save the model.

            save_index: Save the index content. Defaults to True.

            compression: Compress index data. Defaults to True.

            overwrite: Overwrite previous model. Defaults to True.

            include_optimizer: Save optimizer state. Defaults to True.

            signatures: Signatures to save with the model. Defaults to None.

            options: A `tf.saved_model.SaveOptions` to save with the model.
            Defaults to None.

            save_traces (optional): When enabled, the SavedModel will
            store the function traces for each layer. This can be disabled,
            so that only the configs of each layer are stored.
            Defaults to True. Disabling this will decrease serialization
            time and reduce file size, but it requires that all
            custom layers/models implement a get_config() method.
        """

        # save trace doesn't exist prior to 2.4 -- asking for it but not
        # using it

        # call underlying keras method to save the mode graph and its weights
        tf.keras.models.save_model(self,
                                   filepath,
                                   overwrite=overwrite,
                                   include_optimizer=include_optimizer,
                                   signatures=signatures,
                                   options=options,
                                   save_traces=save_traces)
        if hasattr(self, '_index') and self._index and save_index:
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

    # We don't need from_config as the index is reloaded separatly.
    # this is kept as a reminder that it was looked into and decided to split
    # the index reloading instead of overloading this method.
    # @classmethod
    # def from_config(cls, config):
    #     return super().from_config(**config)
