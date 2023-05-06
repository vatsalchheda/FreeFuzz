import paddle
real = paddle.rand([5, 5], paddle.float64)
imag = paddle.rand([5, 5], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2 = [arg_2_0,]
res = paddle.fft.fftshift(arg_1,arg_2,)
