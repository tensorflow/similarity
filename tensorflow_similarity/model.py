from collections import defaultdict
from tqdm.auto import tqdm
from tabulate import tabulate

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow_similarity.indexer import Indexer

from .distances import metric_name_canonializer
from .metrics import precision, f1_score


class SimilarityModel(Model):
    """Sub-classing Keras.Model to allow access to the forward pass values for
    efficient metric-learning.
    """
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

        # self.distance_metrics.update_state(distances)

        # FIXME: add our custom metrics and storage of vector here

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def compile(self,
                optimizer='rmsprop',
                distance='auto',
                loss=None,
                metrics=None,
                distance_metrics=None,
                loss_weights=None,
                mapper='memory',
                matcher='hnsw',
                stat_buffer_size=100,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        "Configures the model for training"

        # init calibration
        self.calibration = None

        # Fetching the distance used from the first loss if auto
        if distance == 'auto':
            if isinstance(loss, list):
                metric_loss = loss[0]
            else:
                metric_loss = loss

            try:
                self.distance = metric_loss.distance
            except:  # noqa
                msg = "distance='auto' only works if the first loss is a\
                       metric loss"

                raise ValueError(msg)
        else:
            self.distance = metric_name_canonializer(distance)

        self._index = Indexer(distance=self.distance,
                              table=mapper,
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
        embeddings = self.predict(x)
        data = x if store_data else None
        self._index.batch_add(embeddings, y, data, build=build, verbose=1)

    def index_reset(self):
        "Reinitialize the index"
        self._index.reset()

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

    def calibrate(self, x, y, k=2, verbose=1):
        # FIXME: use bulk lookup when it will be fixed

        # getting the NN distance and testing if they are of the same class
        if verbose:
            pb = tqdm(total=len(x), desc='Finding NN')

        distances = []
        class_matches = []

        # FIXME: batch_lookup
        embeddings = self.predict(x)
        for idx in range(len(x)):
            knn = self._index.single_lookup(embeddings[idx], k=k)
            for nn in knn:
                distances.append(nn['distance'])
                same_class = 1 if nn['label'] == y[idx] else 0
                class_matches.append(same_class)

            if verbose:
                pb.update()

        total_matches = len(class_matches)
        positive_matches = int(tf.math.reduce_sum(class_matches))
        matching_idxes = [int(x) for x in tf.where(class_matches)]

        if verbose:
            print("num positive matches %d/%s\n" %
                  (positive_matches, total_matches))

        # compute the PR curve
        match_rate = 0
        precision_scores = []
        recall_scores = []
        f1_scores = []
        sorted_distance_values = []
        count = 0

        if verbose:
            pb = tqdm(total=len(distances), desc='computing scores')

        idxs = list(tf.argsort(distances))
        for dist_idx in idxs:
            dist_idx = int(dist_idx)  # needed for speed
            distance_value = distances[dist_idx]

            # print(distance_value)
            # # remove distance with self
            # if not round(distance_value, 4):
            #     continue

            count += 1
            if dist_idx in matching_idxes:
                match_rate += 1
            precision = match_rate / count
            recall = float(match_rate / positive_matches)  # not standard
            f1 = f1_score(precision, recall)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            sorted_distance_values.append(distance_value)

            if verbose:
                pb.update()

        # computing normalized threshold
        # FIXME: make it user parametrizable
        metric_rounding = 2
        thresholds = defaultdict(list)
        curr = 100000

        # FIXME make it user configurable
        # normalized cutpoint
        cutpoints = {}
        very_likely = 0.99
        likely = 0.9
        optimistic = 0.5

        num_distances = len(sorted_distance_values)
        if verbose:
            pb = tqdm(total=num_distances, desc='computing threshold')
        for ridx in range(num_distances):
            idx = num_distances - ridx - 1
            # don't round f1 or distance because we need the numerical
            # precision for precise boundary computations
            f1 = f1_scores[idx]
            distance = sorted_distance_values[idx]

            # used for threshold binning -- should be rounded
            precision = round(float(precision_scores[idx]), metric_rounding)
            recall = round(float(recall_scores[idx]), metric_rounding)

            if precision != curr:
                thresholds['precision'].append(precision)
                thresholds['recall'].append(recall)
                thresholds['f1'].append(f1)
                thresholds['distance'].append(distance)
                curr = precision
                if precision >= very_likely:
                    cutpoints['very_likely'] = float(distance)
                elif precision >= likely:
                    cutpoints['likely'] = float(distance)
                elif precision >= optimistic:
                    cutpoints['optimistic'] = float(distance)
            if verbose:
                pb.update()
        pb.close()

        cutpoints['optimal'] = float(thresholds['distance'][tf.math.argmax(
            thresholds['f1'])])

        for v in thresholds.values():
            v.reverse()

        self.calibration = {"thresholds": thresholds, "cutpoints": cutpoints}

        if verbose:
            rows = []
            for k, v in self.calibration['cutpoints'].items():
                rows.append([k, v])
            print(tabulate(rows, headers=['cutpoint', 'distance']))

        return self.calibration

    def match(self, x, cutpoint='optimal', no_match_class=-1, verbose=0):
        """Match a set of examples against the calibrated index

        For the match function to work, the index must be calibrated using
        calibrate().

        Args:
            x (tensor): examples to be matched against the index.
            cutpoint (str, optional): What calibration threshold to use.
            Defaults to 'optimal' which is the optimal F1 threshold computed
            with calibrate().
            no_match_class (int, optional): What class value to assign when
            there is no match. Defaults to -1.

            verbose (int, optional). Defaults to 0.

        Returns:
            list(int): Return which class matches for each supplied example.

        Notes:
            This function matches all the cutpoints at once internally as there
            is little performance downside to do so and allows to do the
            evaluation in a single go.

        """
        # basic checks
        if not self.calibration:
            raise ValueError('Model uncalibrated: run model.calibration()')

        if cutpoint not in self.calibration['cutpoints'] and cutpoint != 'all':
            msg = 'Unknonw cutpoint: %s - available %s' % (
                cutpoint, self.calibration['cutpoints'])
            raise ValueError(msg)

        # get embeddings
        embeddings = self.predict(x)

        # FIXME batch lookup
        if verbose:
            pb = tqdm(total=len(embeddings), desc='Matching up embeddings')

        matches = defaultdict(list)
        for idx in range(len(embeddings)):
            knn = self._index.single_lookup(embeddings[idx], k=1)
            distance = knn[0]['distance']

            # compute match for each cutoff points
            for name, cut_distance in self.calibration['cutpoints'].items():
                if distance < cut_distance:
                    # ! don't remove the cast, otherwise eval get very slow.
                    matches[name].append(int(knn[0]['label']))
                else:
                    matches[name].append(no_match_class)

            if verbose:
                pb.update()

        if verbose:
            pb.close()

        if cutpoint == 'all':  # returns all the cutpoints for eval purpose.
            return matches
        else:  # normal match behavior
            return matches[cutpoint]

    def evaluate_index(self, x, y,  no_match_class=-1, verbose=1):
        """Evaluate model matching accuracy on a given dataset.

        Args:
            x (tensor): Examples to be matched against the index.
            y (tensor): [description]

            no_match_class (int, optional):  What class value to assign when
            there is no match. Defaults to -1.

            verbose (int, optional): Display results if set to 1 otherwise
            results are returned silently. Defaults to 1.

        Returns:
            dict: evaluation metrics
        """

        results = defaultdict(int)
        matches = self.match(x,
                             verbose=verbose,
                             cutpoint='all',
                             no_match_class=no_match_class)

        num_examples = len(x)
        cutpoints = list(self.calibration['cutpoints'].keys())
        y_pred = defaultdict(list)
        y_true = defaultdict(list)

        if verbose:
            pb = tqdm(total=num_examples, desc='Computing')

        for idx in range(num_examples):
            val = int(y[idx])

            y_true['no_cutpoint'].append(val)
            y_pred['no_cutpoint'].append(matches['optimal'][idx])

            for cutpoint in cutpoints:
                pred = matches[cutpoint][idx]
                if pred != no_match_class:
                    y_true[cutpoint].append(val)
                    y_pred[cutpoint].append(pred)

            if verbose:
                pb.update()
        pb.close()

        # computing metrics
        results = defaultdict(dict)
        METRICS_FN = [['precision', precision]]

        if verbose:
            pb = tqdm(total=len(METRICS_FN) + 2, desc="computing metrics")

        # standardized metric
        for m in METRICS_FN:
            for cutpoint in cutpoints:
                tval = float(m[1](y_true[cutpoint], y_pred[cutpoint]))
                results[cutpoint][m[0]] = tval

            if verbose:
                pb.update()

        # Recall
        for cutpoint in cutpoints:
            results[cutpoint]['recall'] = len(y_pred[cutpoint]) / num_examples
        if verbose:
            pb.update()

        # F1
        for cutpoint in cutpoints:
            results[cutpoint]['f1'] = f1_score(results[cutpoint]['precision'],
                                               results[cutpoint]['recall'])
        if verbose:
            pb.update()

        if verbose:
            pb.close()

        # display results if verbose
        if verbose:
            rows = []
            for metric_name in results['optimal'].keys():
                row = [metric_name]
                for cutpoint in cutpoints:
                    row.append(results[cutpoint][metric_name])
                rows.append(row)
            print(tabulate(rows, headers=['metric'] + cutpoints))

        return results
