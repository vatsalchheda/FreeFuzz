import paddle
arg_1_tensor = paddle.randint(-1,16384,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.acosh(arg_1,)
