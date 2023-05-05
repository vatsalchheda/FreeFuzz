import paddle
arg_1 = 3
arg_class = paddle.fluid.dygraph.nn.InstanceNorm(arg_1,)
arg_2_0_tensor = paddle.rand([1, 3, 1, 2], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
