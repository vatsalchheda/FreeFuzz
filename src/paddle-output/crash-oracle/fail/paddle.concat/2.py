import paddle
arg_1_0_tensor = paddle.randint(-8, 4, [1, 140], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-32, 32, [1, 1], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = -55
res = paddle.concat(arg_1,axis=arg_2,)
