import paddle
arg_1_tensor = paddle.randint(-2048,32768,[0], dtype=paddle.int16)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ifft(arg_1,)
