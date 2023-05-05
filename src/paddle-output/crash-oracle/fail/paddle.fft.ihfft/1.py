import paddle
arg_1_tensor = paddle.rand([8, 8, 9, 8, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -46
arg_3 = False
arg_4 = "backward"
res = paddle.fft.ihfft(arg_1,arg_2,arg_3,arg_4,)
