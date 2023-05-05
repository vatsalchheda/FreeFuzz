import paddle
arg_1_tensor = paddle.randint(-512, 8, [64, 10], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_0 = 32
arg_2 = [arg_2_0,]
arg_3_0 = 0
arg_3 = [arg_3_0,]
arg_4_0 = 0
arg_4 = [arg_4_0,]
res = paddle.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
