import paddle

out1 = paddle.arange(5)
# [0, 1, 2, 3, 4]

out2 = paddle.arange(3, 9, 2.0)
# [3, 5, 7]

# use 4.999 instead of 5.0 to avoid floating point rounding errors
out3 = paddle.arange(4.999, dtype='float32')
# [0., 1., 2., 3., 4.]

start_var = paddle.to_tensor([3])
out4 = paddle.arange(start_var, 7)
# [3, 4, 5, 6]