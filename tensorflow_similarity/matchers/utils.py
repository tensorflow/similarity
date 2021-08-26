from typing import Union

from .classification_match import ClassificationMatch
from .match_majority_vote import MatchMajorityVote
from .match_nearest import MatchNearest


def make_classification_matcher(
        matcher: Union[str, 'ClassificationMatch']) -> 'ClassificationMatch':
    """Convert classification matcher from str name to object if needed.

    Args:
        metric: ClassificationMatch() or matcher name.

    Raises:
        ValueError: matcher name is invalid.

    Returns:
        ClassificationMatch: Instantiated matcher if needed.
    """
    # ! Matcher must be non-instantiated.
    MATCHER_ALIASES = {
        "match_nearest": MatchNearest,
        "match_majority_vote": MatchMajorityVote,
    }

    if isinstance(matcher, str):
        if matcher.lower() in MATCHER_ALIASES:
            matcher = MATCHER_ALIASES[matcher.lower()]()
        else:
            raise ValueError('Unknown matcher name:', matcher, ' typo?')

    return matcher
