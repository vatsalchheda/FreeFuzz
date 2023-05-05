import paddle
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fluid.layers.nn.shape(arg_1,)
