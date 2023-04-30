import paddle
arg_1_tensor = paddle.randint(-256,64,[2, 2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ifft2(arg_1,)
