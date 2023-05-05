import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
arg_4 = "forward"
res = paddle.fft.irfftn(arg_1,arg_2,arg_3,arg_4,)
