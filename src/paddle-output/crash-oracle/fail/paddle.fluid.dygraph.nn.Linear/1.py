import paddle
arg_1 = 71.0
arg_2 = "max"
arg_class = paddle.fluid.dygraph.nn.Linear(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([3, 1024], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
