import paddle
arg_1_tensor = paddle.rand([2, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.log1p(arg_1,)
