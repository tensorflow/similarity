import nmslib
from pathlib import Path

from .matcher import Matcher


class NMSLibMatcher(Matcher):

    def __init__(self, distance='cosine', algorithm='nmslib_hnsw', verbose=1):

        if distance == 'cosine':
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

    def add(self, embedding, idx, build=True, verbose=1):
        self._matcher.addDataPoint(idx, embedding)
        if build:
            self._build(verbose=verbose)

    def batch_add(self, embeddings, idxs, build=True, verbose=1):

        # !addDataPoint and addDataPointBAtch have inverted parameters
        if verbose:
            print('|-Adding embeddings to fast NN matcher index.')
        self._matcher.addDataPointBatch(embeddings, idxs)

        if build:
            self._build()
            if verbose:
                print('|-Optimizing NN matcher index.')

    def lookup(self, embedding, k=5):
        idxs, distances = self._matcher.knnQuery(embedding, k=k)
        return idxs, distances

    def batch_lookup(self, embeddings, k=5):
        batch_idxs = []
        batch_distances = []

        # FIXME: make it parallel or use the batch api
        for emb in embeddings:
            dist, idxs = self._matcher.knnQuery(emb, k=k)
            batch_idxs.append(idxs)
            batch_distances.append(dist)

        return batch_idxs, batch_distances

    def save(self, path):
        fname = self.__make_fname(path)
        self._matcher.saveIndex(fname, save_data=True)

    def load(self, path):
        fname = self.__make_fname(path)
        self._matcher.loadIndex(fname, load_data=True)

    def _build(self, verbose=0):
        """Build the index this is need to take into account the new points
        """
        show = True if verbose else False
        self._matcher.createIndex(print_progress=show)

    def __make_fname(self, path):
        return str(Path(path) / 'index_matcher.bin')
