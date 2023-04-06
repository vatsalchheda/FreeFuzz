import paddle

# input is a Tensor which shape is [3, 4, 5]
input = paddle.rand([3, 4, 5])

[x0, x1, x2] = paddle.unbind(input, axis=0)
# x0.shape [4, 5]
# x1.shape [4, 5]
# x2.shape [4, 5]

[x0, x1, x2, x3] = paddle.unbind(input, axis=1)
# x0.shape [3, 5]
# x1.shape [3, 5]
# x2.shape [3, 5]
# x3.shape [3, 5]