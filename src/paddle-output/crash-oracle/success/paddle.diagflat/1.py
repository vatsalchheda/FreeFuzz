import paddle
arg_1_tensor = paddle.randint(-4096, 32768, [2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.diagflat(arg_1,)
