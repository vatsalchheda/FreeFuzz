import paddle
arg_1_tensor = paddle.rand([64, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.cpu(arg_1,)
