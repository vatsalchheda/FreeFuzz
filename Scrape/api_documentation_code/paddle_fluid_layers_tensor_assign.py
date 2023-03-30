import paddle
import numpy as np
data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
array = np.array([[1, 1],
                  [3, 4],
                  [1, 3]]).astype(np.int64)
result1 = paddle.zeros(shape=[3, 3], dtype='float32')
paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
result3 = paddle.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]