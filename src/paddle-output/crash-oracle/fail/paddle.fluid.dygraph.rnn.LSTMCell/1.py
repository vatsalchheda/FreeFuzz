import paddle
arg_1 = 256
arg_2 = 172
arg_class = paddle.fluid.dygraph.rnn.LSTMCell(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([64, 128], dtype=paddle.float64)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([64, 256], dtype=paddle.float64)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.rand([64, 256], dtype=paddle.float64)
arg_3_2 = arg_3_2_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
res = arg_class(*arg_3)
