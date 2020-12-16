import math
from tensorflow.keras.utils import Sequence
from abc import abstractmethod


class Sampler(Sequence):

    def __init__(self,
                 class_per_batch,
                 batch_size=32,
                 batch_per_epoch=1000,
                 augmenter=None,
                 scheduler=None):

        self.epoch = 0  # track epoch count
        self.class_per_batch = class_per_batch
        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch
        self.augmenter = augmenter
        self.scheduler = scheduler

    @abstractmethod
    def get_examples(self, batch_id, num_classes, example_per_class):
        """Get the set of examples that would be used to create a single batch.

        Notes:
         - before passing the batch data to TF, the sampler will call the
        augmenter function (if any) on the returned example.

         - A batch_size = num_classes * example_per_class

         - This function must be defined in the subclass.

        Args:
            batch_id (int): id of the batch in the epoch.
            num_classes ([type]): How many class should be present in the
            examples.
            example_per_class (int): How many example per class should be
            returned.
        """
        raise NotImplementedError('must be implemented by subclass')

    # [Shared mechanics]
    def __len__(self):
        return self.batch_per_epoch

    def on_epoch_end(self):
        # scheduler -> batch_size, class_per_batch?
        # if self.scheduler:
            # fixme scheduler idea
            # self.scheduler(self.epoch)
        self.epoch += 1

    def __getitem__(self, batch_id):
        return self.generate_batch(batch_id)

    def generate_batch(self, batch_id):

        example_per_class = math.ceil(self.batch_size // self.class_per_batch)
        # ! can't have less than 2 example per class in a batch
        example_per_class = max(example_per_class, 2)

        x, y = self.get_examples(batch_id, self.class_per_batch,
                                 example_per_class)

        # strip an example if need to be. This might happen due to rounding
        if len(x) != self.batch_size:
            x = x[:self.batch_size]
            y = y[:self.batch_size]

        # perform data augmentation
        if self.augmenter:
            x = self.augmenter(x)

        return x, y
