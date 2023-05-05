import paddle
arg_class = paddle.nn.RNNCellBase()
arg_1_0_tensor = paddle.rand([4, 4], dtype=paddle.float64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([0, 1024], dtype=paddle.float64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
