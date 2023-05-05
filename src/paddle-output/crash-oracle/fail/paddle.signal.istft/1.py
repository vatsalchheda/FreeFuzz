import paddle
real = paddle.rand([8, 257, 376], paddle.float64)
imag = paddle.rand([8, 257, 376], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = -16
res = paddle.signal.istft(arg_1,n_fft=arg_2,)
