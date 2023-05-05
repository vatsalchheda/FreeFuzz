import paddle
arg_1_tensor = paddle.rand([3, 10, 3, 7], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fluid.dygraph.base.to_variable(arg_1,)
