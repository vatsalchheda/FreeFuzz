import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([[[4, 8, 1, 9],
                       [7, 2, 0, 9],
                       [6, 9, 2, 6]]], dtype="float32")
w = paddle.to_tensor([[[9, 3, 4],
                       [0, 0, 7],
                       [2, 5, 6]],
                      [[0, 3, 4],
                       [2, 9, 7],
                       [5, 6, 8]]], dtype="float32")

y = F.conv1d(x, w)
print(y)
# Tensor(shape=[1, 2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[133., 238.],
#          [160., 211.]]])