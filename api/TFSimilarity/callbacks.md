<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.callbacks" />
<meta itemprop="path" content="Stable" />
</div>
# Module: TFSimilarity.callbacks
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/callbacks.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



## Classes
[`class Callback`](../TFSimilarity/callbacks/Callback.md): Abstract base class used to build new callbacks.
[`class EvalCallback`](../TFSimilarity/callbacks/EvalCallback.md): Epoch end evaluation callback that build a test index and evaluate
[`class EvalMetric`](../TFSimilarity/callbacks/EvalMetric.md): Helper class that provides a standard way to create an ABC using
[`class FloatTensor`](../TFSimilarity/callbacks/FloatTensor.md): Float tensor 
[`class IntTensor`](../TFSimilarity/callbacks/IntTensor.md): Integer tensor
[`class MemoryEvaluator`](../TFSimilarity/callbacks/MemoryEvaluator.md): In memory index performance evaluation and calibration.
[`class Path`](../TFSimilarity/callbacks/Path.md): PurePath subclass that can make system calls.
[`class SplitValidationLoss`](../TFSimilarity/callbacks/SplitValidationLoss.md): A split validation callback.
[`class Tensor`](../TFSimilarity/callbacks/Tensor.md): The base class of all dense Tensor objects.
## Functions
[`make_metric(...)`](../TFSimilarity/callbacks/make_metric.md): Covert metric from str name to object if needed.
