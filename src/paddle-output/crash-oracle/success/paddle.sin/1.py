import paddle
arg_1_tensor = paddle.rand([5000, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.sin(arg_1,)
