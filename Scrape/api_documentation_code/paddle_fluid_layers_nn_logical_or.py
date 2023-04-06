import paddle
import numpy as np

x_data = np.array([True, False], dtype=np.bool_).reshape(2, 1)
y_data = np.array([True, False, True, False], dtype=np.bool_).reshape(2, 2)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
res = paddle.logical_or(x, y)
print(res) # [[ True  True] [ True False]]