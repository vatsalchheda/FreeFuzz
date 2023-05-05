import paddle
arg_1 = 16
arg_2 = -1024
arg_class = paddle.nn.LSTMCell(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([2, 16], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
