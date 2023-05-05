import paddle
arg_1_tensor = paddle.rand([4, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.gradient(arg_1,)
