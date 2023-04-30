import paddle
arg_1_tensor = paddle.randint(-8,16,[0, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
res = paddle.linalg.cholesky(arg_1,upper=arg_2,)
