import paddle
arg_1 = False
arg_2 = None
arg_3 = None
arg_4 = 0
arg_class = paddle.fluid.initializer.XavierInitializer(uniform=arg_1,fan_in=arg_2,fan_out=arg_3,seed=arg_4,)
arg_5_0_tensor = paddle.rand([128], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_5_1 = arg_5_1_tensor.clone()
arg_5 = [arg_5_0,arg_5_1,]
res = arg_class(*arg_5)
