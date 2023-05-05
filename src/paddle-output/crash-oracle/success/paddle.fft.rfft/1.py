import paddle
arg_1_tensor = paddle.rand([8, 5, 6, 9, 6], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = -1
arg_4 = "backward"
res = paddle.fft.rfft(arg_1,arg_2,arg_3,arg_4,)
