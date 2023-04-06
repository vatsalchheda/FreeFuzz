import paddle
# x is a 2-D Tensor:
x = paddle.to_tensor([[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]])
out1 = paddle.count_nonzero(x)
# [3]
out2 = paddle.count_nonzero(x, axis=0)
# [0, 1, 2]
out3 = paddle.count_nonzero(x, axis=0, keepdim=True)
# [[0, 1, 2]]
out4 = paddle.count_nonzero(x, axis=1)
# [2, 1, 0]
out5 = paddle.count_nonzero(x, axis=1, keepdim=True)
#[[2],
# [1],
# [0]]

# y is a 3-D Tensor:
y = paddle.to_tensor([[[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]],
                      [[0., 2.5, 2.6], [0., 0., 2.4], [2.1, 2.2, 2.3]]])
out6 = paddle.count_nonzero(y, axis=[1, 2])
# [3, 6]
out7 = paddle.count_nonzero(y, axis=[0, 1])
# [1, 3, 5]