import paddle
arg_1_tensor = paddle.randint(-512,256,[2, 3], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.hfft2(arg_1,)
