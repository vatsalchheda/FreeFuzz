import paddle
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = inf
res = paddle.linalg.norm(arg_1,p=arg_2,)
