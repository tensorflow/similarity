from typing import Dict, Type, Union

from .classification_match import ClassificationMatch
from .match_majority_vote import MatchMajorityVote
from .match_nearest import MatchNearest


def make_classification_matcher(
        matcher: Union[str, ClassificationMatch]) -> ClassificationMatch:
    """Convert classification matcher from str name to object if needed.

    Args:
        matcher: {'match_nearest', 'match_majority_vote'} or
        ClassificationMatch object. Defines the classification matching,
        e.g., match_nearest will count a True Positive if the query_label
        is equal to the label of the nearest neighbor and the distance is
        less than or equal to the distance threshold.

    Raises:
        ValueError: matcher name is invalid.

    Returns:
        ClassificationMatch: Instantiated matcher if needed.
    """
    # ! Matcher must be non-instantiated.
    MATCHER_ALIASES: Dict[str, Type[ClassificationMatch]] = {
        "match_nearest": MatchNearest,
        "match_majority_vote": MatchMajorityVote,
    }

    if isinstance(matcher, str):
        if matcher.lower() in MATCHER_ALIASES:
            matcher = MATCHER_ALIASES[matcher.lower()]()
        else:
            raise ValueError(f'Unknown matcher name: {matcher}, typo?')

    return matcher
