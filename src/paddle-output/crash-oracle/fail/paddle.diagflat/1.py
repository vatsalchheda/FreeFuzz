import paddle
arg_1_tensor = paddle.randint(-2048, 2, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.diagflat(arg_1,offset=arg_2,)
