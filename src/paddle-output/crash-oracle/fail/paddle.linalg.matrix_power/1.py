import paddle
arg_1_tensor = paddle.rand([3, 1024], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -55
res = paddle.linalg.matrix_power(arg_1,arg_2,)
