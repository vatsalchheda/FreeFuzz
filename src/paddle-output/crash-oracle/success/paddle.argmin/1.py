import paddle
arg_1_tensor = paddle.randint(-8,1,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.argmin(arg_1,axis=arg_2,)
