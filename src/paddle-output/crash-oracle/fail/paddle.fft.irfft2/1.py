import paddle
arg_1_tensor = paddle.randint(-16,1024,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.irfft2(arg_1,)
