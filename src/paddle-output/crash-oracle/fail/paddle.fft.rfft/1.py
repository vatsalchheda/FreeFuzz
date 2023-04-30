import paddle
arg_1_tensor = paddle.randint(-4,2,[4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.rfft(arg_1,)
