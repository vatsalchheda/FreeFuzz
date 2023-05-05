import paddle
arg_1_tensor = paddle.randint(-1024,16384,[1, 46], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.cumsum(arg_1,axis=arg_2,)
