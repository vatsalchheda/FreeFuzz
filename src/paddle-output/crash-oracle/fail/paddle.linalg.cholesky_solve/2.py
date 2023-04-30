import paddle
arg_1_tensor = paddle.randint(-1024,8,[3, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,8192,[3, 3], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = False
res = paddle.linalg.cholesky_solve(arg_1,arg_2,upper=arg_3,)
