import paddle
arg_1_tensor = paddle.randint(-1, 128, [8], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.unique_consecutive(arg_1,)
