import paddle
arg_1 = -34.999
arg_class = paddle.fluid.regularizer.L2DecayRegularizer(arg_1,)
arg_2_0_tensor = paddle.rand([10, 10], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([10, 10], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2_2 = arg_2_2_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = arg_class(*arg_2)
