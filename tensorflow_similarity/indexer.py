"Index embedding to allow distance based lookup"

import nmslib
from .metrics import metric_name_canonializer
from .mappers import MemoryMapper
from tqdm.auto import tqdm


class Indexer():

    def __init__(self,
                 distance='cosine',
                 metadata={},
                 mapper='memory',
                 method='hnsw'):

        self.metadata = metadata

        # translate metric name to nmslib distance
        self.distance = metric_name_canonializer(distance)
        if self.distance == 'cosine':
            self.space_name = 'cosinesimil'
        else:
            raise ValueError('Unsupported metric space')

        # mapper id > data
        if mapper == 'memory':
            self.mapper = MemoryMapper()
        else:
            self.mapper = mapper

        self.index = nmslib.init(method=method, space=self.space_name)

    def add(self, embedding, label=None, data=None):
        """ Add a single embedding to the indexer

        Args:
            embedding (tensor): TF similarity model output / embeddings.
            label (str/int, optional): label(s) associated with the
            embedding. Defaults to None.
            data (Tensor, optional): input data associated with the embedding.
            Defaults to None.
        """
        # store data and get its id
        idx = self._store_data(embedding, label, data)

        # add index to the embedding
        # !the order of parameters between addDataPoint and addDataPointBatch
        # !are inverted
        self.index.addDataPoint(idx, embedding)

    def batch_add(self,
                  embeddings,
                  labels=None,
                  data=None,
                  verbose=1):
        """Add a batch of embeddings to the indexer

        Args:
            embeddings (list(tensor)): TF similarity model output / embeddings.
            labels (list(str/int), optional): label(s) associated with the
            embedding. Defaults to None.
            datas (list(Tensor), optional): input data associated with the
            embedding. Defaults to None.
            verbose (bool, optional): display progress bar as
            indexing is taking place. Defaults to 1.
        """

        # store points
        batch = []
        batch_idxs = []

        if verbose:
            pb = tqdm(total=len(embeddings),
                      desc='Storing embeddings',
                      unit='embedings')

        for idx in range(len(embeddings)):
            emb = embeddings[idx]
            lbl = labels[idx] if not isinstance(labels, type(None)) else None
            dta = data[idx] if not isinstance(data, type(None)) else None

            # store the point
            idx = self._store_data(emb, lbl, dta)

            # build batch
            batch.append(emb)
            batch_idxs.append(idx)

            if verbose:
                pb.update(1)

        # add point to the index
        self.index.addDataPointBatch(batch, batch_idxs)

        if verbose:
            pb.close()

    def build(self, verbose=1):
        """Build the indexer
        """
        self.index.createIndex(print_progress=verbose)

    def lookup(self, embedding, k=5):
        """Find the k closest match of a given embedding

        Args:
            embedding ([type]): [description]
            k (int, optional): [description]. Defaults to 5.
        Returns
            list: list of k nearest matched embeddings.
        """

        results = []
        idxs, distances = self.index.knnQuery(embedding, k=k)
        for i, idx in enumerate(idxs):
            data = self.mapper.get(idx)
            data['distance'] = distances[i]
            results.append(data)
        return results

    def batch_lookup(self, embedding, k=5):
        """Find the k closest match of a batch of embeddings

        Args:
            embeddings ([type]): [description]
            k (int, optional): [description]. Defaults to 5.
            threads (int, optional). Defaults to 4
        Returns
            list: list of k nearest matched embeddings.
        """
        raise NotImplementedError('WIP')

    def save(self):
        raise NotImplementedError('WIP')
        info = {
            'distance': self.distance,
            'space_name': self.space_name,
            'batch_size': self.batch_size,
            'metadata': self.metadata
        }

        # serialize index
        # serialize mapping

        return info

    def load(self):
        raise NotImplementedError('WIP')

    def _store_data(self, embedding, label, data):
        "store data using mapper and assign it an id"
        data = {"embedding": embedding, 'label': label, 'data': data}
        return self.mapper.add(data)
