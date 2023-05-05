import paddle
arg_1_tensor = paddle.randint(-2, 64, [1, 1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.fluid.layers.nn.reduce_all(arg_1,)
