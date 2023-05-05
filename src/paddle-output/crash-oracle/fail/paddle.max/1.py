import paddle
arg_1_tensor = paddle.randint(-2, 32, [100], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.max(arg_1,)
