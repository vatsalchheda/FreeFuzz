import paddle
arg_1_tensor = paddle.randint(-8192,16384,[1, 30000], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -62.0
res = paddle.full_like(arg_1,arg_2,)
