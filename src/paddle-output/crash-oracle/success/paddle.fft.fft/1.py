import paddle
arg_1_tensor = paddle.rand([7, 7, 2, 6, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = 3
arg_4 = "backward"
res = paddle.fft.fft(arg_1,arg_2,arg_3,arg_4,)
