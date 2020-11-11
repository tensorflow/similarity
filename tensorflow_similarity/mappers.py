class MemoryMapper():

    def __init__(self):
        self.num_references = 0
        self.htable = {}  # potentially use an array if we don't include a del.

    def add(self, data):
        """Add a value to the mapper

        Args:
            data (dict): data to store in the mapper.

        Returns:
            int: associated id
        """
        idx = self.num_references
        self.htable[idx] = data
        self.num_references += 1
        return idx

    def get(self, idx):
        """Get the data from

        Args:
            idx (int): id value to lookup

        Returns:
            dict: data associated with the id
        """
        return self.htable[idx]

    def delete(self, idx):
        """Remove point from the mapper"""
        del self.data[idx]
