from collections import defaultdict
from tqdm.auto import tqdm
from tabulate import tabulate

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow_similarity.indexer import Indexer

from .distances import metric_name_canonializer
from .metrics import precision, accuracy


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
                              mapper=mapper,
                              matcher=matcher,
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
            precision = match_rate / (count)
            recall = float(match_rate / positive_matches)
            f1 = (precision * recall / (precision + recall)) * 2

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
        # normalized labels
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
                    cutpoints['very_likely'] = distance
                elif precision >= likely:
                    cutpoints['likely'] = distance
                elif precision >= optimistic:
                    cutpoints['optimistic'] = distance
            if verbose:
                pb.update()
        pb.close()

        cutpoints['optimal'] = thresholds['distance'][tf.math.argmax(
            thresholds['f1'])]

        for v in thresholds.values():
            v.reverse()

        self.calibration = {"thresholds": thresholds, "cutpoints": cutpoints}

        if verbose:
            rows = []
            for k, v in self.calibration['cutpoints'].items():
                rows.append([k, v])
            print(tabulate(rows, headers=['cutpoint', 'distance']))

        return self.calibration

    def match(self, x, threshold='optimal', no_match_class=-1, verbose=0):
        """Match a set of examples against the calibrated index

        For the match function to work, the index must be calibrated using
        calibrate().

        Args:
            x (tensor): input to be matched against the index
            cutpoint (str, optional): What calibration threshold to use.
            Defaults to 'optimal' which is the optimal F1 threshold computed with
            calibrate().
            no_match_class (int, optional): What class value to assign when
            there is no match. Defaults to -1.

            verbose (int, optional). Defaults to 0.

        Returns:
            list(int): Return which class matches for each supplied example.
        """
        matches = []

        # basic checks
        if not self.calibration:
            raise ValueError('Model uncalibrated: run model.calibration()')

        if threshold not in self.calibration['cutpoints']:
            msg = 'Unknonw cutpoint: %s - available %s' % (
                threshold, self.calibration['cutpoints'])
            raise ValueError(msg)

        # FIXME batch lookup
        embeddings = self.predict(x)

        if verbose:
            pb = tqdm(total=len(embeddings), desc='Matching up embeddings')

        for idx in range(len(x)):
            knn = self._index.single_lookup(embeddings[idx], k=1)
            distance = knn[0]['distance']
            if distance < self.calibration['cutpoints'][threshold]:
                matches.append(knn[0]['label'])
            else:
                matches.append(no_match_class)

            if verbose:
                pb.update()

        if verbose:
            pb.close()

        return matches

    def evaluate_index(self, x, y, verbose=1):

        results = defaultdict(int)
        matches = self.match(x, verbose=verbose)
        y_pred = []

        # restrict to < threshold
        y_thresholded = []
        y_pred_thresholded = []
        if verbose:
            pb = tqdm(total=len(matches), desc='Computing')

        for idx, match in enumerate(matches):
            val = y[idx]
            match = int(match)  # needed or its going to be horribly slow
            y_pred.append(match)
            if match > 0:
                y_thresholded.append(val)
                y_pred_thresholded.append(match)
                results['matched'] += 1
            else:
                results['unknown'] += 1

            if verbose:
                pb.update()
        pb.close()

        total = len(matches)
        threshold_recall = results['matched'] / total

        METRICS_FN = [['accuracy', accuracy], ['precision', precision]]

        # thresholded metrics
        metrics = {}
        metrics['cutpoint'] = {
            'recall': threshold_recall,
        }
        metrics['full'] = {'recall': 1}

        rows = [['recall', round(threshold_recall, 5), 1]]
        for m in tqdm(METRICS_FN, desc="Computing metrics"):
            # thresholded
            tval = float(m[1](y_thresholded, y_pred_thresholded))
            metrics['cutpoint'][m[0]] = tval

            # full
            fval = float(m[1](y, y_pred))
            metrics['full'][m[0]] = fval

            rows.append([m[0], round(tval, 5), round(fval, 5)])

        print(tabulate(rows, headers=['metric', 'cutpoint <', 'all']))

        return metrics
