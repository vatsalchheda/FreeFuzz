import paddle
real = paddle.rand([2], paddle.float64)
imag = paddle.rand([2], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
res = paddle.tanh(arg_1,)
