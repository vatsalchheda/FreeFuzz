import paddle
real = paddle.rand([8, 6, 5, 8, 7], paddle.float64)
imag = paddle.rand([8, 6, 5, 8, 7], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = -1
arg_4 = "reflect"
res = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
