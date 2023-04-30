import paddle
arg_1_tensor = paddle.randint(-1024,8,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.frac(arg_1,)
