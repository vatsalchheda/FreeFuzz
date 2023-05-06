import paddle
arg_1_tensor = paddle.rand([10, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.mean(arg_1,)
