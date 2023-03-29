import paddle

# the axis is a int element
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]])
out1 = paddle.prod(x)
# [0.0002268]

out2 = paddle.prod(x, -1)
# [0.027  0.0084]

out3 = paddle.prod(x, 0)
# [0.02 0.06 0.3  0.63]

out4 = paddle.prod(x, 0, keepdim=True)
# [[0.02 0.06 0.3  0.63]]

out5 = paddle.prod(x, 0, dtype='int64')
# [0 0 0 0]

# the axis is list
y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]])
out6 = paddle.prod(y, [0, 1])
# [105. 384.]

out7 = paddle.prod(y, (1, 2))
# [  24. 1680.]