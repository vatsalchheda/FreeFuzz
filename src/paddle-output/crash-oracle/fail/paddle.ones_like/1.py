import paddle
arg_1_tensor = paddle.randint(-8192,16384,[1, 64, 295], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.ones_like(arg_1,)
