import paddle
arg_1_tensor = paddle.rand([2, 0], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.nn.elementwise_pow(arg_1,arg_2,)
