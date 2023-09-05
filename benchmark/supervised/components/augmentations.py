from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from tensorflow.image import random_flip_left_right, random_flip_up_down

from tensorflow_similarity.augmenters.augmentation_utils.cropping import (
    center_crop,
    crop_and_resize,
)

AUGMENTATIONS = {}
AUGMENTATIONS["random_resized_crop"] = lambda p: partial(
    crop_and_resize,
    height=p.get("height", 277),
    width=p.get("width", 277),
    area_range=p.get("area_range", (0.2, 1.0)),
)
AUGMENTATIONS["random_flip"] = (
    lambda p: random_flip_left_right if p.get("mode", "horizontal") == "horizontal" else random_flip_up_down
)
AUGMENTATIONS["center_crop"] = lambda p: partial(
    center_crop,
    height=p.get("height", 256),
    width=p.get("width", 256),
    crop_proportion=p.get("crop_proportion", 1.0),
)


# TODO(ovallis): Return type should be tuple[Callable[[FloatTensor], FloatTensor]], but
# mypy doesn't recogonize the return types of the callabels.
def make_augmentations(cfg: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple([AUGMENTATIONS[aug_id](params) for aug_id, params in cfg.items()])
