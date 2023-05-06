import paddle
arg_1_tensor = paddle.rand([5, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = False
arg_3_1 = 18.0
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = None
res = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
