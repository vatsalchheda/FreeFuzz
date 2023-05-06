import paddle
arg_1_tensor = paddle.rand([0], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.round(arg_1,)
