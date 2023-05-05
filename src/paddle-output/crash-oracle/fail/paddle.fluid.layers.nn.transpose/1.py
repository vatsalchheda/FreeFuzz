import paddle
arg_1_tensor = paddle.rand([4, -1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 0
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.fluid.layers.nn.transpose(arg_1,arg_2,)
