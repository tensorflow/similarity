# Module: TFSimilarity.distances
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



## Classes
- [`class ABC`](../TFSimilarity/distances/ABC.md): Helper class that provides a standard way to create an ABC using
- [`class CosineDistance`](../TFSimilarity/distances/CosineDistance.md): Compute pairwise cosine distances between embeddings.
- [`class Distance`](../TFSimilarity/distances/Distance.md): Note: don't forget to add your distance to the DISTANCES list
- [`class EuclideanDistance`](../TFSimilarity/distances/EuclideanDistance.md): Compute pairwise euclidean distances between embeddings.
- [`class FloatTensor`](../TFSimilarity/callbacks/FloatTensor.md): Float tensor 
- [`class InnerProductSimilarity`](../TFSimilarity/distances/InnerProductSimilarity.md): Compute the pairwise inner product between embeddings.
- [`class ManhattanDistance`](../TFSimilarity/distances/ManhattanDistance.md): Compute pairwise Manhattan distances between embeddings.
- [`class SquaredEuclideanDistance`](../TFSimilarity/distances/SquaredEuclideanDistance.md): Compute pairwise squared Euclidean distance.
## Functions
- [`distance_canonicalizer(...)`](../TFSimilarity/distance_metrics/distance_canonicalizer.md): Normalize user requested distance to its matching Distance object.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>
<tr>
<td>
DISTANCES<a id="DISTANCES"></a>
</td>
<td>
`[<tensorflow_similarity.distances.InnerProductSimilarity object at 0x7fecf7189a00>,
 <tensorflow_similarity.distances.EuclideanDistance object at 0x7fecf7189b50>,
 <tensorflow_similarity.distances.SquaredEuclideanDistance object at 0x7fecf7189af0>,
 <tensorflow_similarity.distances.ManhattanDistance object at 0x7fecf71971f0>,
 <tensorflow_similarity.distances.CosineDistance object at 0x7fecf71970a0>]`
</td>
</tr>
</table>
