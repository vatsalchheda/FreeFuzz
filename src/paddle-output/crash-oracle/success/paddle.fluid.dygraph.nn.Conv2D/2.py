import paddle
arg_1 = 3
arg_2 = 2
arg_3 = 3
arg_class = paddle.fluid.dygraph.nn.Conv2D(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([10, 3, 32, 79], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = arg_class(*arg_4)
