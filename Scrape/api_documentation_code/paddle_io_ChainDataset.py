import numpy as np
import paddle
from paddle.io import IterableDataset, ChainDataset


# define a random dataset
class RandomDataset(IterableDataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(10):
            image = np.random.random([32]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            yield image, label

dataset = ChainDataset([RandomDataset(10), RandomDataset(10)])
for image, label in iter(dataset):
    print(image, label)