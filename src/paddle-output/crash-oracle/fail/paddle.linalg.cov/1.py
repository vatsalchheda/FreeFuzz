import paddle
arg_1_tensor = paddle.randint(-32768,16384,[2, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.cov(arg_1,)
