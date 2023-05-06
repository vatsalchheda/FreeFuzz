import paddle
arg_1_tensor = paddle.rand([1, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.floor(arg_1,)
