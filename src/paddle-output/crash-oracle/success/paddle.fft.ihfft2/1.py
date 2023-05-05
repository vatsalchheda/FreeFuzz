import paddle
arg_1_tensor = paddle.rand([5, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ihfft2(arg_1,)
