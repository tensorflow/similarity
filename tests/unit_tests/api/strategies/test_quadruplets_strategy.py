# Copyright 2020 Google LLC
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

import os
import pytest
import tempfile
import numpy as np

from tensorflow_similarity.api.engine.similarity_model import SimilarityModel
from tensorflow_similarity.api.strategies.quadruplet_strategy import (
    HardQuadrupletLossStrategy, QuadrupletLossStrategy)
from tensorflow_similarity.api.tasks.utils_for_test import (
    gen_learnable_testdata, learnable_model)
from tensorflow.keras.models import load_model


def test_create():
    x, y = gen_learnable_testdata()
    print(len(x["intinput"]), len(x["intinputv"]), len(y))
    model = learnable_model()
    model = QuadrupletLossStrategy(model)

    model.fit(x, y, prewarm_epochs=0, epochs=1, verbose=0)


def test_serialize(tmpdir):
    model_path = os.path.join(str(tmpdir), "serialized_model.h5")

    x, y = gen_learnable_testdata()
    model = learnable_model()
    model = QuadrupletLossStrategy(model)

    model.save(model_path)

    model2 = load_model(model_path)

    assert isinstance(model2, type(model))
    assert model.get_config() == model2.get_config()
    simple_fields = ["hard_mining", "tmp_dir", "strategy", "name"]
    for field in simple_fields:
        f = getattr(model, field)
        f2 = getattr(model2, field)

        assert (f == f2,
                "Mismatch in %s - expected '%s' and received '%s'" % (
                    field, f, f2))

    out = model.predict(x)
    out2 = model2.predict(x)

    assert np.allclose(out, out2)


def test_create_prewarm_sanitycheck():
    """ Setting a prewarm_epochs without having auxillary tasks is an error."""
    x, y = gen_learnable_testdata()
    model = learnable_model()
    model = QuadrupletLossStrategy(model)

    with pytest.raises(ValueError):
        model.fit(x, y, prewarm_epochs=1, epochs=1, verbose=0)


if __name__ == '__main__':
    from pathlib import Path
    test_serialize(Path("/tmp"))
