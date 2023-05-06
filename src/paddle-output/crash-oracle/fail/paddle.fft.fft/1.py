import paddle
arg_1_tensor = paddle.rand([7, 2, 4, 8, 8], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -54
arg_2_1 = -1
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = -19
arg_4 = "backward"
res = paddle.fft.fft(arg_1,arg_2,arg_3,arg_4,)
