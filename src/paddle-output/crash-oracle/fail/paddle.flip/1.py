import paddle
arg_1_tensor = paddle.randint(-8192, 256, [3, 2, 2], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2_1 = 1
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.flip(arg_1,arg_2,)
