import paddle
arg_1_tensor = paddle.randint(-16, 16, [4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 31
arg_2 = [arg_2_0,]
res = paddle.fluid.layers.nn.unsqueeze(arg_1,arg_2,)
