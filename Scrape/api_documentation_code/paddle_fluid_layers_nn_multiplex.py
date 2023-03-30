import paddle
import numpy as np
img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
inputs = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
index = paddle.to_tensor(np.array([[1], [0]]).astype(np.int32))
res = paddle.multiplex(inputs, index)
print(res) # [array([[5., 6.], [3., 4.]], dtype=float32)]