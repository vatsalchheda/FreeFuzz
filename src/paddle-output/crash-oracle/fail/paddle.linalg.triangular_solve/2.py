import paddle
arg_1_tensor = paddle.randint(-2048,8192,[3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,256,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = "sum"
res = paddle.linalg.triangular_solve(arg_1,arg_2,upper=arg_3,)
