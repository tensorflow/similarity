# Copyright 2021 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
