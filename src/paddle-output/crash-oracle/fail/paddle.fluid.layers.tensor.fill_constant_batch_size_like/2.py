import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2 = [arg_2_0,]
arg_3 = "paddleVarType"
arg_4 = 0.0
res = paddle.fluid.layers.tensor.fill_constant_batch_size_like(arg_1,arg_2,arg_3,arg_4,)
