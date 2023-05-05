import paddle
arg_1_tensor = paddle.randint(-2048, 16, [4, 4], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 2
res = paddle.fluid.layers.nn.elementwise_floordiv(arg_1,arg_2,)
