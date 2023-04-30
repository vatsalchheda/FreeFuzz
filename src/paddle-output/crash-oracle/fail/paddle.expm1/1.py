import paddle
arg_1_tensor = paddle.randint(-512,16384,[2, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.expm1(arg_1,)
