from tabulate import tabulate
from tqdm.auto import tqdm
from .evaluator import Evaluator
from collections import defaultdict
import tensorflow as tf


class MemoryEvaluator(Evaluator):
    "In memory evaluator system"

    def __init__(self, targets={}, cutpoints={}, thresholds={}):

        self.targets = targets
        self.cutpoints = cutpoints
        self.thresholds = thresholds

    def is_calibrated(self):
        return True if 'optimal' in self.cutpoints else False

    def calibrate(self, y, labels, distances, targets, verbose=1):

        # set new threshold targets
        self.targets = targets

        # Compute PR metrics for all points while sorting distances
        # we use y_preds, y_true to be able to reuse standardized functions
        num_distances = len(distances)
        sorted_distance_values = []
        scores = defaultdict(list)

        if verbose:
            pb = tqdm(total=num_distances, desc='computing scores')

        # ! casting is needed for speed
        positive_matches = 0
        total_matches = 0
        distance_idxs = [int(i) for i in list(tf.argsort(distances))]
        for idx in distance_idxs:

            if labels[idx] == y[idx]:
                positive_matches += 1
            total_matches += 1

            # metrics
            precision = positive_matches / total_matches
            recall = total_matches / num_distances  # non-standard
            f1_score = 2 * (precision * recall) / (precision + recall)

            scores['precision'].append(precision)
            scores['recall'].append(recall)
            scores['f1_score'].append(f1_score)

            # store sorted distance
            sorted_distance_values.append(distances[idx])

            if verbose:
                pb.update()

        if verbose:
            pb.close()

        # sfind the thresholds by going from left to right
        # FIXME: make the number of thresholds user parametrizable?
        rounding = 2
        thresholds = defaultdict(list)
        prev_precision = 100000
        num_distances = len(sorted_distance_values)

        if verbose:
            pb = tqdm(total=num_distances, desc='computing thresholds')

        for ridx in range(num_distances):
            # Note: unsure if going left to right or right to left if better
            # if going right to left, don't forget to reverse the thresholds
            # idx = num_distances - ridx - 1  # reversed
            idx = ridx  # not reversing

            # used for threshold binning -- should be rounded
            curr_precision = round(scores['precision'][idx], rounding)

            if curr_precision != prev_precision:

                # used for threshold binning -- should be rounded
                thresholds['precision'].append(curr_precision)
                thresholds['recall'].append(
                    round(scores['recall'][idx], rounding))

                # don't round f1 or distance because we need the numerical
                # precision for precise boundary computations
                curr_distance = sorted_distance_values[idx]
                thresholds['distance'].append(curr_distance)
                thresholds['f1_score'].append(scores['f1_score'][idx])

                # update current precision threshold
                prev_precision = curr_precision

                # test if the precision meet any our precision target
                for target_name, target_value in self.targets.items():
                    if curr_precision >= target_value:
                        self.cutpoints[target_name] = curr_distance

            if verbose:
                pb.update()

        if verbose:
            pb.close()

        # record tresholds
        self.thresholds = thresholds

        # find the optimal cutpoint
        max_f1_idx = tf.math.argmax(thresholds['f1_score'])
        self.cutpoints['optimal'] = float(thresholds['distance'][max_f1_idx])

        # reverse the threshold values so they go from left to right as user
        # expect
        # for v in thresholds.values():
        #     v.reverse()

        # display result if asked
        if verbose:
            rows = [[k, v] for k, v in self.cutpoints.items()]
            print(tabulate(rows, headers=['cutpoints', 'distance']))

        # mark calibration as completed
        self._is_calibrated = True

        return self.to_config()

    def match(self, distances, labels, no_match_label=-1, verbose=1):
        """Match distances against the various cutpoints thresholds

        Note:
        Args:
            distances (list(float)): the distances to match againt the
            cutpoints.

            labels(list(int)): the associated labels with the distance.

            no_match_label (int, optional): What label value to assign when
            there is no match. Defaults to -1.

            verbose (int): display progression. Default to 1.

        Notes:
            1. its up to the model code to decide which of the cutpoints to
            use / show to the users. The evaluator returns all of them as there
            are little performance downside and makes code clearer and simpler.

            2. the function is responsible to return the list of class matched
            to allows implementation to use additional criterias if they choose
            to.

        Returns:
            dict: {cutpoint: list(bool)}
        """
        if verbose:
            pb = tqdm(total=len(distances) * len(self.cutpoints),
                      desc='matching embeddings')

        matches = defaultdict(list)
        for name, threshold in self.cutpoints.items():
            for idx, distance in enumerate(distances):
                if distance <= threshold:
                    label = labels[idx]
                else:
                    label = no_match_label
                matches[name].append(label)

                if verbose:
                    pb.update()

        if verbose:
            pb.close()

        return matches

    @staticmethod
    def from_config(config):
        return MemoryEvaluator(config['targets'], config['cutpoints'],
                               config['thresholds'])

    def to_config(self):
        return {
            "targets": self.targets,
            "cutpoints": self.cutpoints,
            "thresholds": self.thresholds,
        }
