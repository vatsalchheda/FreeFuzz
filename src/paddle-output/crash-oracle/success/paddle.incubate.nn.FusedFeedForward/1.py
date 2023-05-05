import paddle
arg_1 = 8
arg_2 = 13
arg_class = paddle.incubate.nn.FusedFeedForward(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([1, 8, 8], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
