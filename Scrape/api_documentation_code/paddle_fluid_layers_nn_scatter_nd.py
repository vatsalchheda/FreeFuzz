import paddle
import numpy as np

index_data = np.array([[1, 1],
                       [0, 1],
                       [1, 3]]).astype(np.int64)
index = paddle.to_tensor(index_data)
updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
shape = [3, 5, 9, 10]

output = paddle.scatter_nd(index, updates, shape)