import paddle
arg_1_tensor = paddle.randint(-8192,32768,[2, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 56
res = paddle.squeeze(arg_1,axis=arg_2,)
