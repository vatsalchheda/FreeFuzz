import paddle
arg_1_tensor = paddle.rand([3, 1, 6, 9, 9], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
arg_4 = "ortho"
res = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,)
