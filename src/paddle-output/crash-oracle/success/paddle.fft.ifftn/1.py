import paddle
arg_1_tensor = paddle.rand([5, 5, 7, 8, 9], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
arg_4 = "forward"
res = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,)
