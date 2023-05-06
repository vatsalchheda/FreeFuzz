import paddle
arg_1_tensor = paddle.rand([1, 6, 1, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.uniform_(arg_1,)
