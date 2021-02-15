import math
from copy import copy
from tqdm.auto import tqdm
from .evaluator import Evaluator
from collections import defaultdict
import tensorflow as tf
from typing import DefaultDict, List, Dict, Union
from tensorflow_similarity.metrics import EvalMetric, make_metrics


class MemoryEvaluator(Evaluator):
    "In memory evaluator system"

    def evaluate(self,
                 index_size: int,
                 metrics: List[Union[str, EvalMetric]],
                 targets_labels: List[int],
                 lookups: List[List[Dict[str, Union[float, int]]]],
                 distance_rounding: float = 8
                 ) -> Dict[str, Union[float, int]]:

        # [nn[{'distance': xxx}, ]]
        # normalize metrics
        eval_metrics: List[EvalMetric] = make_metrics(metrics)

        # get max_k from first lookup result
        max_k = len(lookups[0])

        # ! don't add intermediate computation that don't speedup multiple
        # ! metrics. Those goes into the metrics themselves.
        # compute intermediate representations used by metrics
        # rank 0 == no match / distance 0 == unknown
        num_matched = 0
        match_ranks = [0] * len(targets_labels)
        match_distances = [0.0] * len(targets_labels)
        for lidx, lookup in enumerate(lookups):
            true_label = targets_labels[lidx]
            for nidx, n in enumerate(lookup):
                rank = nidx + 1
                if n['label'] == true_label:
                    # print(n['label'], true_label, lookup)
                    match_ranks[lidx] = rank
                    match_distances[lidx] = round(n['distance'],
                                                  distance_rounding)
                    num_matched += 1
        num_unmatched = len(targets_labels) - num_matched

        # compute metrics
        evaluation = {}
        for m in eval_metrics:
            evaluation[m.name] = m.compute(
                max_k,
                targets_labels,
                num_matched,
                num_unmatched,
                index_size,
                match_ranks,
                match_distances,
                lookups,  # e.g used when k > 1 for confusion matrix
            )

        return evaluation

    def calibrate(self,
                  index_size: int,
                  calibration_metric: EvalMetric,
                  thresholds_targets: Dict[str, float],
                  targets_labels: List[int],
                  lookups: List[List[Dict[str, Union[float, int]]]],
                  extra_metrics: List[Union[str, EvalMetric]] = [],
                  distance_rounding: int = 8,
                  metric_rounding: int = 6,
                  verbose: int = 1):

        # distance are rounded because of numerical instablity
        # copy threshold targets as we are going to delete them and don't want
        # to alter users supplied data
        thresholds_targets = copy(thresholds_targets)

        # making a single list of metrics
        combined_metrics = list(extra_metrics)  # covariance problem
        combined_metrics.append(calibration_metric)

        # Distance preparation
        # flattening
        distances = []
        for l in lookups:
            for n in l:
                distances.append(round(n['distance'], distance_rounding))

        # sorting them
        # !keep the casting to int() or it will be awefully slow
        sorted_distances_idxs = [int(i) for i in list(tf.argsort(distances))]
        sorted_distances_values = [distances[i] for i in sorted_distances_idxs]
        num_distances = len(distances)

        # evaluating performance as distance value increase
        evaluations = []
        if verbose:
            pb = tqdm(total=num_distances, desc='Evaluating')

        for dist in sorted_distances_values:
            # update distance theshold for metrics
            for m in combined_metrics:
                m.distance_threshold = dist

            res = self.evaluate(index_size, combined_metrics, targets_labels,
                                lookups, distance_rounding)
            res['distance'] = dist
            evaluations.append(res)
            if verbose:
                pb.update()

        if verbose:
            pb.close()

        # find the thresholds by going from right to left

        # which direction metric improvement is?
        #! loop is right to left so max is decreasing and min is increasing
        if calibration_metric.direction == 'max':
            # we want the lowest value at the largest distance possible
            cmp = self._is_lower
            prev_value = math.inf  # python 3.x only
        else:
            # we want the highest value at the largest distance possible
            cmp = self._is_higher
            prev_value = 0

        # we need a collection of list to apply vectorize operations and make
        # the analysis / viz of the calibration data signifcantly easier
        thresholds: DefaultDict[str, List[Union[int, float]]] = defaultdict(list)  # noqa
        cutpoints: DefaultDict[str, Dict[str, Union[str, float, int]]] = defaultdict(dict)  # noqa
        num_distances = len(sorted_distances_values)
        if verbose:
            pb = tqdm(total=num_distances, desc='computing thresholds')

        # looping from right to left as we want the max distance for a given
        # metric value
        for ridx in range(num_distances):
            idx = num_distances - ridx - 1  # reversed

            # Rounding the calibration metric to create bins
            curr_eval = evaluations[idx]
            calibration_value = curr_eval[calibration_metric.name]
            curr_value = round(calibration_value, metric_rounding)

            # ? if bug use this line check that the values evolve correclty.
            # print(curr_value, prev_value, cmp(curr_value, prev_value))

            if cmp(curr_value, prev_value):

                # add a new distance threshold
                thresholds['value'].append(curr_value)

                # ! the correct distance is already in the eval data
                # record the value for all the metrics requested by the user
                for k, v in curr_eval.items():
                    thresholds[k].append(v)

                # update current threshold value
                prev_value = curr_value

                # check if the current value meet or exceed threshold target
                to_delete = []  # can't delete in an interation loop
                for name, value in thresholds_targets.items():
                    if cmp(curr_value, value, equal=True):
                        cutpoints[name] = {'name': name}  # useful for display
                        for k in thresholds.keys():
                            cutpoints[name][k] = thresholds[k][-1]
                        to_delete.append(name)

                # removing found targets to avoid finding lower value
                # recall we go from right to left in the evaluation
                for name in to_delete:
                    del thresholds_targets[name]

            if verbose:
                pb.update()

        if verbose:
            pb.close()

        # find the optimal cutpoint
        if calibration_metric.direction == 'min':
            best_idx = tf.math.argmin(thresholds[calibration_metric.name])
        else:
            best_idx = tf.math.argmax(thresholds[calibration_metric.name])

        # record its value
        cutpoints['optimal'] = {'name': 'optimal'}  # useful for display
        for k in thresholds.keys():
            cutpoints['optimal'][k] = thresholds[k][best_idx]

        # reverse the threshold so they go from left to right as user expect
        for k in thresholds.keys():  # this syntax is need for mypy ...
            thresholds[k].reverse()

        return thresholds, cutpoints

    def _is_lower(self, curr, prev, equal=False):
        if equal:
            return curr <= prev
        return curr < prev

    def _is_higher(self, curr, prev, equal=False):
        if equal:
            return curr >= prev
        return curr > prev
