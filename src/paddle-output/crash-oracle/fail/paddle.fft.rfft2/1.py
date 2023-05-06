import paddle
arg_1_tensor = paddle.randint(-16, 4, [1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_0 = 4
arg_2_1 = 4
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 45
arg_3_1 = -55
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
res = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
