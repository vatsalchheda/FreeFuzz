import paddle
arg_1_tensor = paddle.rand([2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
arg_3 = True
res = paddle.fluid.layers.nn.reduce_sum(arg_1,dim=arg_2,keep_dim=arg_3,)
