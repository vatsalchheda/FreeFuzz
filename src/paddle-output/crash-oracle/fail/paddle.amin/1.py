import paddle
arg_1_tensor = paddle.randint(-4096,4,[2, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -39
res = paddle.amin(arg_1,axis=arg_2,)
