import paddle
arg_1_tensor = paddle.randint(-16,16384,[7], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.fft(arg_1,)
