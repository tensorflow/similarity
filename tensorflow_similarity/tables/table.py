from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from tensorflow_similarity.types import FloatTensor, PandasDataFrame, Tensor


class Table(ABC):

    @abstractmethod
    def add(self, embedding: FloatTensor,
            label: Optional[int] = None,
            data: Optional[Tensor] = None) -> int:
        """Add a record to the table

        Args:
            embedding (FloatTensor): Record an embedding predicted
            by the model.

            label (int, optional): Class numerical id. Defaults to None.

            data (FloatTensor, optional): Record data. Defaults to None.

        Returns:
            int: associated record id.
        """

    @abstractmethod
    def batch_add(
            self,
            embeddings: List[FloatTensor],
            labels: List[Optional[int]] = None,
            data: List[Optional[Tensor]] = None) -> List[int]:
        """Add a set of record to the table

        Args:
            embeddings (FloatTensor): Record the embeddings predicted
            by the model.

            labels (list(int), optional): Class numerical id. Defaults to None.

            datas (list(FloatTensor), optional): Record data.
            Defaults to None.

        See:
            add() for what a record contains.

        Returns:
            list(int): list of associated record id.
        """

    @abstractmethod
    def get(self, idx: int) -> Tuple[FloatTensor,
                                     Optional[int],
                                     Optional[Tensor]]:
        """Get record from the table

        Args:
            idx (int): lookup record id to fetch.

        Returns:
            record: record associated with the requested record id.
        """

    @abstractmethod
    def batch_get(self, idxs: List[int]
                  ) -> Tuple[List[FloatTensor], List[Optional[int]],
                             List[Optional[Tensor]]]:
        """Get records from the table

        Args:
            idxs (List[int]): lookups record ids to fetch.

        Returns:
            Tuple(List): data associated with the requested record ids.
        """

    @abstractmethod
    def size(self) -> int:
        "Number of record in the table"

    @abstractmethod
    def save(self, path: str, compression: bool = True) -> None:
        """Serializes index on disks

        Args:
            path (str): where to store the data.
            compression (bool): Compress index data. Defaults to True.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """load index on disk

        Args:
            path (str): where to store the data
        """

    @abstractmethod
    def to_data_frame(self, num_items: int = 0) -> PandasDataFrame:
        """Export data as pandas dataframe

        Args:
            num_items (int, optional): Num items to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            pd.DataFrame: a pandas dataframe.
        """
