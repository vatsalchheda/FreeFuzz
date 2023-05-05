import paddle
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 1], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = 1e+20
res = paddle.linalg.triangular_solve(arg_1,arg_2,upper=arg_3,)
