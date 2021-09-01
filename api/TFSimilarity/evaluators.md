# Module: TFSimilarity.evaluators





Evaluates search index performance and calibrates it.


## Use

Evaluators are used for two primary purposes:

- Evaluate model performance on a reference index during training and
evaluation phase via the `evaluate_classification()` and `evaluate_retrieval()`
methods. Evaluation on a reference index is
required to be able to assess model performance using
- [Classification metrics](../classification_metrics/) and
- [Retrieval metrics](../retrieval_metrics/).
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

## Classes

- [`class Evaluator`](../TFSimilarity/callbacks/Evaluator.md): Evaluates search index performance and calibrates it.

- [`class MemoryEvaluator`](../TFSimilarity/callbacks/MemoryEvaluator.md): In memory index performance evaluation and classification.

