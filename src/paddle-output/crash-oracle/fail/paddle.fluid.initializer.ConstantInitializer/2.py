import paddle
arg_1 = 0.0
arg_2 = False
arg_class = paddle.fluid.initializer.ConstantInitializer(value=arg_1,force_cpu=arg_2,)
arg_3_0_tensor = paddle.rand([1], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
