import paddle
arg_1_tensor = paddle.randint(-16384, 1, [10, 4, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.tensor.assign(arg_1,arg_2,)
