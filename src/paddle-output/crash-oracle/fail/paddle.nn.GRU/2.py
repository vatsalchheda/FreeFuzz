import paddle
arg_1 = 16
arg_2 = 31
arg_3 = 2
arg_class = paddle.nn.GRU(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([4, 23], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([2, 4], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
res = arg_class(*arg_4)
