import paddle
arg_1_tensor = paddle.randint(-32768,1,[2, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.nanmean(arg_1,axis=arg_2,)
