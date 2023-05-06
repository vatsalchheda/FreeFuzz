import paddle
real = paddle.rand([4, 4, 4], paddle.float64)
imag = paddle.rand([4, 4, 4], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -2
arg_3_1 = -1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "forward"
arg_5_0 = 1024
arg_5_1 = -16
arg_5 = [arg_5_0,arg_5_1,]
res = paddle.fft.irfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
