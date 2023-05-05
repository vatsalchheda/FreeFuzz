import paddle
arg_1 = 16
arg_2 = 38
arg_3 = "max"
arg_class = paddle.nn.LSTM(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([4, 23, 16, 1], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([63, 2], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
res = arg_class(*arg_4)
