import paddle
arg_1_tensor = paddle.rand([7, 1, 5, 7, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -47
arg_3_1 = -1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
res = paddle.fft.ihfft2(arg_1,arg_2,arg_3,arg_4,)
