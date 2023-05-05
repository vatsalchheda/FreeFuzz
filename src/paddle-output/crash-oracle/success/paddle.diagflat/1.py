import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.diagflat(arg_1,offset=arg_2,)
