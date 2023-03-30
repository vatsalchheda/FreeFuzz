import paddle

x = paddle.ones([3, 2, 4])
paddle.moveaxis(x, [0, 1], [1, 2]).shape
# [4, 3, 2]

x = paddle.ones([2, 3])
paddle.moveaxis(x, 0, 1).shape # equivalent to paddle.t(x)
# [3, 2]