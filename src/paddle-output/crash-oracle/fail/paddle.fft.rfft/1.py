import paddle
arg_1_tensor = paddle.rand([8, 9, 1, 1, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = True
arg_4 = "ortho"
res = paddle.fft.rfft(arg_1,arg_2,arg_3,arg_4,)
