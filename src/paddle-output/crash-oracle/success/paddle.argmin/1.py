import paddle
arg_1_tensor = paddle.randint(-1024, 512, [3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.argmin(arg_1,axis=arg_2,)
