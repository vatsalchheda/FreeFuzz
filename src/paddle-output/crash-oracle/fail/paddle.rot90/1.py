import paddle
arg_1_tensor = paddle.randint(-1, 4096, [2, 2], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = -18
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.rot90(arg_1,arg_2,arg_3,)
