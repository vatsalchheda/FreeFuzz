import paddle
arg_1 = 0.0
arg_2 = 0.4714045207910317
arg_3 = 20
arg_class = paddle.fluid.initializer.NormalInitializer(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.randint(-128,16,[3, 1, 3, 3], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.randint(-32,1024,[2, 2], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
res = arg_class(*arg_4)
