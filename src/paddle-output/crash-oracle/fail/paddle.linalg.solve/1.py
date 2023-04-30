import paddle
arg_1_tensor = paddle.randint(-8192,8,[2, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,4096,[2], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.linalg.solve(arg_1,arg_2,)
