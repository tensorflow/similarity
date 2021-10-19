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
"""Evaluates search index performance and calibrates it.

## Use

Evaluators are used for two primary purposes:

- Evaluate model performance on a reference index during training and
evaluation phase via the `evaluate_classification()` and `evaluate_retrieval()`
methods. Evaluation on a reference index is
required to be able to assess model performance using
[Classification metrics](../classification_metrics/) and
[Retrieval metrics](../retrieval_metrics/).
Those metrics can't be computed without indexing data and looking up
nearest neighbors.

- Calibrating the model requires evaluating various distance thresholds
to find the maximal distance threshold. Those thresholds either meet,
if possible, the user supplied `thresholds_targets` performance value or
the optimal value with respect to the calibration `classification metric`.
Calibration is required to perform matching
because the optimal distance thresholds will change depending on
the model, dataset and, training. Accordingly those thresholds
need to be evaluated empirically for every use-case.
"""
from .memory_evaluator import MemoryEvaluator  # noqa
from .evaluator import Evaluator  # noqa
