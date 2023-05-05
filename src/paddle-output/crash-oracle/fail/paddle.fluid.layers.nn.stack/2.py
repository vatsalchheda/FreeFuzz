import paddle
arg_1_0_tensor = paddle.randint(-2048, 4, [4, 4], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8, 32, [4, 4], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 1033
res = paddle.fluid.layers.nn.stack(arg_1,axis=arg_2,)
