import paddle
arg_1_tensor = paddle.rand([6, 6, 9, 5, 9], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
arg_4 = "backward"
res = paddle.fft.ihfft2(arg_1,arg_2,arg_3,arg_4,)
