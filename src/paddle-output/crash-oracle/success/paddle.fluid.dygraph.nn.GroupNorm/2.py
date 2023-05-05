import paddle
arg_1 = 32
arg_2 = 4
arg_class = paddle.fluid.dygraph.nn.GroupNorm(channels=arg_1,groups=arg_2,)
arg_3_0_tensor = paddle.rand([8, 32, 32], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
