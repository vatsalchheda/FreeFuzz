import numpy as np

from paddle.io import Dataset, DistributedBatchSampler

# init with dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([784]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

dataset = RandomDataset(100)
sampler = DistributedBatchSampler(dataset, batch_size=64)

for data in sampler:
    # do something
    break