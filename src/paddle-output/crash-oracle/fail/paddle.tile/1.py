import paddle
arg_1_tensor = paddle.randint(-16,8192,[4, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 16
arg_2_1 = -3
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.tile(arg_1,arg_2,)
