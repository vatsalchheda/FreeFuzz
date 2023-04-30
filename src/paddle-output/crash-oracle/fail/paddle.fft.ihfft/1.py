import paddle
arg_1_tensor = paddle.randint(-1,64,[6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ihfft(arg_1,)
