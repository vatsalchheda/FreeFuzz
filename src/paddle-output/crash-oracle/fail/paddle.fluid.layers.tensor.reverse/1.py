import paddle
arg_1_tensor = paddle.rand([4, 23, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -19
res = paddle.fluid.layers.tensor.reverse(arg_1,axis=arg_2,)
