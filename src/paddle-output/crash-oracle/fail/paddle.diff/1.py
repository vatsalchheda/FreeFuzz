import paddle
arg_1_tensor = paddle.randint(-2048, 1, [2, 3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.diff(arg_1,axis=arg_2,)
