import paddle
arg_1 = 0.0
arg_2 = 0.02
arg_3 = -7.0
arg_class = paddle.fluid.initializer.TruncatedNormalInitializer(loc=arg_1,scale=arg_2,seed=arg_3,)
arg_4_0_tensor = paddle.rand([3010], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([37, 2], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
res = arg_class(*arg_4)
