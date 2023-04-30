import paddle
arg_1_tensor = paddle.randint(-2,16384,[3, 3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.slogdet(arg_1,)
