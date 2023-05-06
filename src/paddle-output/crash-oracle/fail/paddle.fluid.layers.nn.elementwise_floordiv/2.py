import paddle
arg_1_tensor = paddle.rand([2, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2, 8, [1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.nn.elementwise_floordiv(arg_1,arg_2,)
