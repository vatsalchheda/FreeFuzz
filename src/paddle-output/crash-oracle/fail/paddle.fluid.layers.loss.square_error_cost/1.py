import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128, 256, [64, 1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.loss.square_error_cost(arg_1,arg_2,)
