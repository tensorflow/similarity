from typing import Sequence

import numpy as np

from tensorflow_similarity.types import Lookup


def unpack_lookups(lookups: Sequence[Sequence[Lookup]],
                   attribute: str) -> np.ndarray:
    """Unpack an attribute from a collection Lookup results.

    Args:
        lookups: A 2D collection of Lookup results where the jth row is the k
        neighbors for the jth query.

        attribute: The Lookup attribute to unpack.

    Returns:
        A 2D np.ndarray representing the unpacked attribute.
    """
    all_values = []
    for lu in lookups:
        values = []
        for neighbor in lu:
            values.append(getattr(neighbor, attribute))
        all_values.append(values)
    return np.array(all_values)


def unpack_lookup_labels(lookups):
    return unpack_lookups(lookups, 'label')


def unpack_lookup_distances(lookups):
    return unpack_lookups(lookups, 'distance')


def compute_match_mask(query_labels: np.ndarray,
                       lookup_labels: np.ndarray) -> np.ndarray:
    """Compute a boolean mask (indicator function) marking the TPs in the results.

    Args:
        query_labels: A 1D array of the labels associated with the queries.

        lookup_labels: A 2D array where the jth row is the labels associated
        with the set of k neighbors for the jth query.

    Returns:
        A 2D boolean array indicating which lookups match the label of their
        associated query.
    """
    if query_labels.ndim == 1:
        query_labels = query_labels.reshape(-1, 1)

    match_mask = np.zeros_like(lookup_labels, dtype=float)
    match_mask[np.where(np.equal(lookup_labels, query_labels))] = 1.0

    return match_mask
