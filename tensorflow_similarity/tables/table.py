from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from tensorflow_similarity.types import FloatTensor, PandasDataFrame, Tensor


class Table(ABC):

    @abstractmethod
    def add(self,
            embedding: FloatTensor,
            label: Optional[int] = None,
            data: Optional[Tensor] = None) -> int:
        """Add an Embedding record to the table.

        Args:
            embedding: Embedding predicted by the model.

            label: Class numerical id. Defaults to None.

            data: Data associated with the embedding. Defaults to None.

        Returns:
            Associated record id.
        """
    @abstractmethod
    def batch_add(self,
                  embeddings: List[FloatTensor],
                  labels: List[Optional[int]] = None,
                  data: List[Optional[Tensor]] = None) -> List[int]:
        """Add a set of embedding records to the table.

        Args:
            embeddings: Embeddings predicted by the model.

            labels: Class numerical ids. Defaults to None.

            data: Data associated with the embeddings. Defaults to None.

        See:
            add() for what a record contains.

        Returns:
            List of associated record id.
        """

    @abstractmethod
    def get(self, idx: int) -> Tuple[FloatTensor,
                                     Optional[int],
                                     Optional[Tensor]]:
        """Get an embedding record from the table

        Args:
            idx: Id of the record to fetch.

        Returns:
            record associated with the requested id.
        """

    @abstractmethod
    def batch_get(self, idxs: List[int]
                  ) -> Tuple[List[FloatTensor], List[Optional[int]],
                             List[Optional[Tensor]]]:
        """Get embedding records from the table

        Args:
            idxs: ids of the records to fetch.

        Returns:
            List of records associated with the requested ids.
        """

    @abstractmethod
    def size(self) -> int:
        "Number of record in the table."

    @abstractmethod
    def save(self, path: str, compression: bool = True) -> None:
        """Serializes index on disk.

        Args:
            path: Directory where to store the data.
            compression: Compress index data. Defaults to True.
        """

    @abstractmethod
    def load(self, path: str) -> int:
        """Load index on disk

        Args:
            path: where to store the data

        Returns:
           Number of records reloaded.
        """

    @abstractmethod
    def to_data_frame(self, num_records: int = 0) -> PandasDataFrame:
        """Export data as a Pandas dataframe.

        Args:
            num_records: Number of records to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            pd.DataFrame: a pandas dataframe.
        """
