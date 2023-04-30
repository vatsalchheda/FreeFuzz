import paddle
arg_1_tensor = paddle.randint(-128,32768,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -32
arg_2_1 = -37
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.tile(arg_1,repeat_times=arg_2,)
