import paddle
arg_1_tensor = paddle.rand([1, 64, 294], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.ones_like(arg_1,)
