import paddle
arg_1_tensor = paddle.randint(-1,64,[10, 4, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2 = [arg_2_0,]
res = paddle.fluid.layers.nn.squeeze(arg_1,arg_2,)
