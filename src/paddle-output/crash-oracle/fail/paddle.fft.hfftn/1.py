import paddle
arg_1_tensor = paddle.randint(-1024,128,[3], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.hfftn(arg_1,)
