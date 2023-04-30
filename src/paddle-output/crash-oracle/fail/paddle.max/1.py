import paddle
arg_1_tensor = paddle.randint(-64,32,[2, 2, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -16
arg_2_1 = -16
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.max(arg_1,axis=arg_2,)
