import paddle
arg_1_tensor = paddle.rand([2, 3, 6, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.frac(arg_1,)
