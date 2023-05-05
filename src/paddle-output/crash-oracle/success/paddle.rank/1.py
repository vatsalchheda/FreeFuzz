import paddle
arg_1_tensor = paddle.rand([3, 100, 95, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.rank(arg_1,)
