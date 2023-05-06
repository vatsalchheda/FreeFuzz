import paddle
real = paddle.rand([4, 4, 4], paddle.float64)
imag = paddle.rand([4, 4, 4], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 57.0
arg_3_1 = False
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = "input"
res = paddle.fft.hfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
