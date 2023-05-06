import paddle
arg_1_tensor = paddle.rand([2, 2, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -60
arg_2_1 = 31
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.amin(arg_1,axis=arg_2,)
