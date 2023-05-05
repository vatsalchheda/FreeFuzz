import paddle
arg_1_tensor = paddle.rand([-1, -1, 16], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fluid.layers.nn.shape(arg_1,)
