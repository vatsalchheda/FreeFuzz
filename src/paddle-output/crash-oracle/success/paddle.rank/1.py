import paddle
arg_1_tensor = paddle.rand([3, 100, 100], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.rank(arg_1,)
