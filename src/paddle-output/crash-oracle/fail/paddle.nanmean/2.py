import paddle
arg_1_tensor = paddle.randint(-2,8192,[2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.nanmean(arg_1,axis=arg_2,)
