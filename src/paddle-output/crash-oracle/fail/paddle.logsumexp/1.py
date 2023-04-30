import paddle
arg_1_tensor = paddle.randint(-4,256,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.logsumexp(arg_1,)
