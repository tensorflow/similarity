# Overview

Evaluators are used to compute index matching performance against
a set of [EvalMetrics](../eval_metric.md).

## Uses

Evaluators are used for two primary purposes:

- Evaluate model performance on a reference index during training and
evaluation phase via the `evaluate()` method. Evaluation on a reference index is
required to assess metric learning model performance (e.g accuracy/recall/precision)
Performance metrics, contrary to traditional models, can't be computed without
indexing data and looking up nearest neighboors.

- Calibrate the model which means evaluating which distance thresholds should be
used to meet (if possible) the request target performance. This calibration is
required as depending on the model and dataset, the distance cut-off value
will changes and it need to be evaluated empirically.

## Modules


[EvaluatorTable](memory.md): Default Evaluator implemented as an in-memory
process.Allows fast evaluation that scale well up to a few millions points.

[Table](evaluator.md): Abstract Evaluator class.
