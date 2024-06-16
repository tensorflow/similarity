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

import itertools
import random
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from tensorflow_similarity.visualization.projector_v2 import renderer


def _get_renderer():
    # In Colab, the `google.colab` module is available, but the shell
    # returned by `IPython.get_ipython` does not have a `get_trait`
    # method.
    try:
        import google.colab  # noqa: F401
        import IPython
    except ImportError:
        pass
    else:
        if IPython.get_ipython() is not None:
            # We'll assume that we're in a Colab notebook context.
            raise NotImplementedError("Colab support not implemented")

    # In an IPython command line shell or Jupyter notebook, we can
    # directly query whether we're in a notebook c    ontext.
    try:
        import IPython
    except ImportError:
        pass
    else:
        ipython = IPython.get_ipython()
        if ipython is not None and ipython.has_trait("kernel"):
            return renderer.IPythonRenderer()

    # Otherwise, we're not in a known notebook context.
    raise NotImplementedError("Must use the tool under a notebook context.")


def embedding(
    embeddings: Sequence[Union[Tuple[float, float], Tuple[float, float, float]]],
    labels: Optional[Sequence[Union[str, int]]] = None,
    image_labels: Optional[Sequence[Union[str, int]]] = None,
):
    """ """
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    cur_renderer = _get_renderer()
    handle = cur_renderer.display()
    cur_renderer.send_message(
        handle,
        "update",
        embeddings,
        {"step": 0, "maxStep": 0},
    )

    if labels:
        label_to_color = dict()
        metadata = []
        for label, image in itertools.zip_longest(
            labels, image_labels or [], fillvalue=""
        ):
            color = label_to_color.get(label, None)
            if color is None:
                red = random.randrange(0, 0xFF)
                green = random.randrange(0, 0xFF)
                blue = random.randrange(0, 0xFF)
                color = red * 2 ** 16 + green * 2 ** 8 + blue
                label_to_color.setdefault(label, color)

            print(label, color)
            metadata.append(
                {
                    "label": label,
                    "color": color,
                    "imageLabel": image,
                }
            )

        cur_renderer.send_message(
            handle,
            "meta",
            metadata,
            {"algo": "custom"},
        )
