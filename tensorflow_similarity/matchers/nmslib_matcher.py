import nmslib
from pathlib import Path
from typing import List, Tuple

from tensorflow_similarity.distances import distance_canonicalizer
from .matcher import Matcher
from tensorflow_similarity.types import FloatTensor


class NMSLibMatcher(Matcher):
    """
    Efficiently find nearest embeddings by indexing known embeddings and make
    them searchable using the  [Approximate Nearest Neigboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
    search library [NMSLIB](https://github.com/nmslib/nmslib).
    """

    def __init__(self,
                 distance: str = 'cosine',
                 algorithm: str = 'nmslib_hnsw',
                 verbose: int = 1):

        distance = distance_canonicalizer(distance)

        if distance.name == 'cosine':
            space = 'cosinesimil'
        else:
            raise ValueError('Unsupported metric space')

        if algorithm == 'nmslib_hnsw':
            method = 'hnsw'
        else:
            raise ValueError('Unsupported algorithm')

        show_prog = True if verbose else False

        self._matcher = nmslib.init(method=method, space=space)
        self._matcher.createIndex(print_progress=show_prog)

    def add(self,
            embedding: FloatTensor,
            idx: int,
            build: bool = True,
            verbose: int = 1):
        """Add an embedding to the index

        Args:
            embedding: The embedding to index as computed by
            the similarity model.

            idx: Embedding id as in the index table.
            Returned with the embedding to allow to lookup
            the data associated with a given embedding.

            build: Rebuild the index after the addition.
            Required to make the embedding searchable.
            Set to false to save time between successive addition.
            Defaults to True.

            verbose: Be verbose. Defaults to 1.
        """
        self._matcher.addDataPoint(idx, embedding)
        if build:
            self._build(verbose=verbose)

    def batch_add(self,
                  embeddings: FloatTensor,
                  idxs: List[int],
                  build: bool = True,
                  verbose: int = 1):
        """Add a batch of embeddings to the matcher.

        Args:
            embeddings: List of embeddings to add to the index.

            idxs (int): Embedding ids as in the index table. Returned with
            the embeddings to allow to lookup the data associated
            with the returned embeddings.

            build: Rebuild the index after the addition. Required to
            make the embeddings searchable. Set to false to save
            time between successive addition. Defaults to True.

            verbose: Be verbose. Defaults to 1.
        """
        # !addDataPoint and addDataPointBAtch have inverted parameters
        if verbose:
            print('|-Adding embeddings to fast NN matcher index.')
        self._matcher.addDataPointBatch(embeddings, idxs)

        if build:
            self._build()
            if verbose:
                print('|-Optimizing NN matcher index.')

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

        # FIXME: make it parallel or use the batch api
        for emb in embeddings:
            dist, idxs = self._matcher.knnQuery(emb, k=k)
            batch_idxs.append(idxs)
            batch_distances.append(dist)

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
