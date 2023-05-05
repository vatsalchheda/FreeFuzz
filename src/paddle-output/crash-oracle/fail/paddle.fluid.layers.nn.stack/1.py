import paddle
arg_1_0_tensor = paddle.randint(-64, 64, [4, 4], dtype=paddle.int64arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-32, 16384, [4, 4], dtype=paddle.int64arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = None
res = paddle.fluid.layers.nn.stack(arg_1,axis=arg_2,)
