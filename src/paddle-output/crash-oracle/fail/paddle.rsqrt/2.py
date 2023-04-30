import paddle
arg_1_tensor = paddle.randint(-16,16384,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.rsqrt(arg_1,)
