import paddle
arg_1_tensor = paddle.rand([3, 4, 4, 3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 16
arg_2_1 = -40
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 1
arg_3_1 = 2
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
res = paddle.fft.rfftn(arg_1,arg_2,arg_3,arg_4,)
