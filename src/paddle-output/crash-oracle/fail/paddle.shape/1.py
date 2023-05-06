import paddle
arg_1_tensor = paddle.randint(-8, 2048, [4, 1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.shape(arg_1,)
