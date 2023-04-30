import paddle
arg_1_tensor = paddle.randint(-512,8,[3], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.hfft(arg_1,)
