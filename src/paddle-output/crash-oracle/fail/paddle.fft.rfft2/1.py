import paddle
arg_1_tensor = paddle.randint(-2,8192,[5, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.rfft2(arg_1,)
