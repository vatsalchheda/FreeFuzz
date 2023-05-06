import paddle
arg_1_tensor = paddle.rand([18, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.square(arg_1,)
