from os import name
import nmslib
from pathlib import Path
from typing import List, Tuple, Union

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from .matcher import Matcher
from tensorflow_similarity.types import FloatTensor


class NMSLibMatcher(Matcher):
    """
    Efficiently find nearest embeddings by indexing known embeddings and make
    them searchable using the  [Approximate Nearest Neigboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
    search library [NMSLIB](https://github.com/nmslib/nmslib).
    """

    def __init__(self,
                 distance: Union[Distance, str],
                 dims: int,
                 algorithm: str = 'nmslib_hnsw',
                 **kwargs):

        distance_obj: Distance = distance_canonicalizer(distance)

        # convert to nmslib word
        if distance_obj.name == 'cosine':
            space = 'cosinesimil'
        elif distance_obj.name == 'euclidean':
            space = 'l2'
        elif distance_obj.name == 'manhattan':
            space = 'l1'
        else:
            raise ValueError('Unsupported metric space')

        if algorithm == 'nmslib_hnsw':
            method = 'hnsw'
        else:
            raise ValueError('Unsupported algorithm')

        self._matcher = nmslib.init(method=method, space=space)
        self._matcher.createIndex()


    def add(self,
            embedding: FloatTensor,
            idx: int,
            verbose: int = 1,
            build: bool = True,
            **kwargs):
        """Add an embedding to the index

        Args:
            embedding: The embedding to index as computed by
            the similarity model.

            idx: Embedding id as in the index table.
            Returned with the embedding to allow to lookup
            the data associated with a given embedding.

            verbose: Be verbose. Defaults to 1.

            build: Rebuild the index after the addition.
            Required to make the embedding searchable.
            Set to false to save time between successive addition.
            Defaults to True.

        """
        self._matcher.addDataPoint(idx, embedding)
        if build:
            self._build(verbose=verbose)

    def batch_add(self,
                  embeddings: FloatTensor,
                  idxs: List[int],
                  verbose: int = 1,
                  build: bool = True,
                  **kwargs):
        """Add a batch of embeddings to the matcher.

        Args:
            embeddings: List of embeddings to add to the index.

            idxs (int): Embedding ids as in the index table. Returned with
            the embeddings to allow to lookup the data associated
            with the returned embeddings.

            verbose: Be verbose. Defaults to 1.

            build: Rebuild the index after the addition. Required to
            make the embeddings searchable. Set to false to save
            time between successive addition. Defaults to True.
        """
        # !addDataPoint and addDataPointBAtch have inverted parameters
        if verbose:
            print('|-Adding embeddings to index.')
        self._matcher.addDataPointBatch(embeddings, idxs)

        if build:
            if verbose:
                print('|-Building index.')
            self._build(verbose=verbose)

    def lookup(self,
               embedding: FloatTensor,
               k: int = 5) -> Tuple[List[int], List[float]]:
        """Find embedding K nearest neighboors embeddings.

        Args:
            embedding: Query embedding as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """
        idxs: List[int] = []
        distances: List[float] = []
        idxs, distances = self._matcher.knnQuery(embedding, k=k)
        return idxs, distances

    def batch_lookup(self,
                     embeddings: FloatTensor,
                     k: int = 5) -> Tuple[List[List[int]], List[List[float]]]:
        """Find embeddings K nearest neighboors embeddings.

        Args:
            embedding: Batch of query embeddings as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """
        batch_idxs = []
        batch_distances = []

        nn = self._matcher.knnQueryBatch(embeddings, k=k)
        for n in nn:
            batch_idxs.append(n[0])
            batch_distances.append(n[1])
        return batch_idxs, batch_distances

    def save(self, path: str):
        """Serializes the index data on disk

        Args:
            path: where to store the data
        """
        fname = self.__make_fname(path)
        self._matcher.saveIndex(fname, save_data=True)

    def load(self, path: str):
        """load index on disk

        Args:
            path: where to store the data
        """
        fname = self.__make_fname(path)
        self._matcher.loadIndex(fname, load_data=True)

    def _build(self, verbose=0):
        """Build the index this is need to take into account the new points
        """
        show = True if verbose else False
        self._matcher.createIndex(print_progress=show)

    def __make_fname(self, path):
        return str(Path(path) / 'index_matcher.bin')
