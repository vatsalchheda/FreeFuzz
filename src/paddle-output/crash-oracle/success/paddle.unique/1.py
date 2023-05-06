import paddle
arg_1_tensor = paddle.randint(-1024, 64, [6], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.unique(arg_1,)
