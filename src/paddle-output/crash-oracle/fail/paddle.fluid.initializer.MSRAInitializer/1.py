import paddle
arg_1 = False
arg_class = paddle.fluid.initializer.MSRAInitializer(uniform=arg_1,)
arg_2_0_tensor = paddle.randint(-64,2048,[1024, 10], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-1024,1024,[2, 2], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
res = arg_class(*arg_2)
