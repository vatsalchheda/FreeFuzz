import numpy as np
import paddle
from paddle.io import Dataset, ComposeDataset


# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([32]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

dataset = ComposeDataset([RandomDataset(10), RandomDataset(10)])
for i in range(len(dataset)):
    image1, label1, image2, label2 = dataset[i]
    print(image1)
    print(label1)
    print(image2)
    print(label2)