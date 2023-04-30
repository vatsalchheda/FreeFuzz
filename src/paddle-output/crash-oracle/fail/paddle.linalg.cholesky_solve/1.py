import paddle
arg_1_tensor = paddle.randint(-32768,64,[3, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,8192,[3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = True
res = paddle.linalg.cholesky_solve(arg_1,arg_2,upper=arg_3,)
