import paddle
arg_1_tensor = paddle.rand([2, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.amin(arg_1,axis=arg_2,)
