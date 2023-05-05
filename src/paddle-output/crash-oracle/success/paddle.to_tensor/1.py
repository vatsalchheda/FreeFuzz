import paddle
arg_1_tensor = paddle.rand([4, 8, 4, 9], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.to_tensor(arg_1,)
