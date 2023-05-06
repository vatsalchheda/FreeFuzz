import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -2
res = paddle.linalg.matrix_power(arg_1,arg_2,)
