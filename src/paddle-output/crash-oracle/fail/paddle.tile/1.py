import paddle
arg_1_tensor = paddle.randint(-64, 2, [3], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.tile(arg_1,repeat_times=arg_2,)
