import paddle
arg_1_tensor = paddle.rand([3, 1, 3, 2, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 11
arg_3 = -6
arg_4 = "backward"
res = paddle.fft.ihfft(arg_1,arg_2,arg_3,arg_4,)
