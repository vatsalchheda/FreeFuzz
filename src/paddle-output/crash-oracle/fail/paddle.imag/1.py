import paddle
arg_1_tensor = paddle.randint(-4096,16384,[3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.imag(arg_1,)
