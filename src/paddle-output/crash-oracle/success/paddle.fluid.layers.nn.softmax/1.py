import paddle
arg_1_tensor = paddle.rand([4, 4, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fluid.layers.nn.softmax(arg_1,)
