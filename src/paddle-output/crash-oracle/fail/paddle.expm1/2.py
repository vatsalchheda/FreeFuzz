import paddle
arg_1_tensor = paddle.randint(-512,8192,[2, 1, 12, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.expm1(arg_1,)
