import paddle
arg_1_tensor = paddle.randint(-128, 8192, [2, 0, 1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.lgamma(arg_1,)
