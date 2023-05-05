import paddle
arg_1 = "sum"
arg_2 = "replicate"
arg_3 = 0
arg_class = paddle.fluid.initializer.NormalInitializer(loc=arg_1,scale=arg_2,seed=arg_3,)
arg_4_0_tensor = paddle.rand([32, 32, 1], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
res = arg_class(*arg_4)
