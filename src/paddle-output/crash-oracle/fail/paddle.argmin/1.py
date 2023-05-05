import paddle
arg_1_tensor = paddle.randint(-8192, 64, [3, 4], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.argmin(arg_1,axis=arg_2,)
