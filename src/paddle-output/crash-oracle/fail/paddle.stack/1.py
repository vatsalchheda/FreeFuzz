import paddle
arg_1_tensor = paddle.rand([2, 4, 8, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.stack(arg_1,)
