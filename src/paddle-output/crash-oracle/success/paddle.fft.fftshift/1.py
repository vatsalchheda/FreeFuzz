import paddle
real = paddle.rand([5, 5], paddle.float64)
imag = paddle.rand([5, 5], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
res = paddle.fft.fftshift(arg_1,arg_2,)
