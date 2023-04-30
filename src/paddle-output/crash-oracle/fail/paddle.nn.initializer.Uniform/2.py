import paddle
arg_1 = -0.17677669529663687
arg_2 = 0.17677669529663687
arg_class = paddle.nn.initializer.Uniform(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-1024,512,[128, 32], dtype=paddle.float64)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(0,1,[2, 2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
