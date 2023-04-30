import paddle
arg_class = paddle.fluid.initializer.XavierInitializer()
arg_1_0_tensor = paddle.randint(-512,1,[784, 200], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-256,32,[2, 2], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
