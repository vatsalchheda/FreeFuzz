import paddle
arg_1_tensor = paddle.randint(-2, 16, [1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.fluid.layers.tensor.concat(arg_1,arg_2,)
