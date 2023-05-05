import paddle
arg_1_tensor = paddle.rand([16], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ifftshift(arg_1,)
