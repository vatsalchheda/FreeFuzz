import paddle
real = paddle.rand([3, 3, 7, 5, 4], paddle.float64)
imag = paddle.rand([3, 3, 7, 5, 4], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = -36
arg_4 = "backward"
res = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,)
