import paddle
arg_1_tensor = paddle.rand([64, 16, 10, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.erf(arg_1,)
