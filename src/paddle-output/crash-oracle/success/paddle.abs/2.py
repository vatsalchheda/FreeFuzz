import paddle
arg_1_tensor = paddle.randint(-1024, 2048, [5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.abs(arg_1,)
