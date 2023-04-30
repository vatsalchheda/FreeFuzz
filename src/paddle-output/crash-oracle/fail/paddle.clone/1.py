import paddle
arg_1_tensor = paddle.randint(-1,16384,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.clone(arg_1,)
