class MemoryMapper():

    def __init__(self):

        # We are using an array for scalability and speed.
        # Makes del a massive pain but its infrequent
        self.htable = []
        self.num_items = 0

    def add(self, data):
        """Add a value to the mapper

        Args:
            data (dict): data to store in the mapper.

        Returns:
            int: associated id
        """
        idx = self.num_items
        self.htable.append(data)
        self.num_items += 1
        return idx

    def get(self, idx):
        """Get the data from

        Args:
            idx (int): id value to lookup

        Returns:
            dict: data associated with the id
        """
        return self.htable[idx]

    def size(self):
        "Number of elements in the mapper"
        return self.num_items

    def delete(self, idx):
        """Remove point from the mapper"""
        raise NotImplementedError
