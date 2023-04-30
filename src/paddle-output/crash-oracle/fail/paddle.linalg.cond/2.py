import paddle
arg_1_tensor = paddle.randint(-4096,16384,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -inf
res = paddle.linalg.cond(arg_1,p=arg_2,)
