import paddle
arg_1_tensor = paddle.rand([8, 8, 9, 8, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.sgn(arg_1,)
