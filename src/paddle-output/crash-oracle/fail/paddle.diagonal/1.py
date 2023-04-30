import paddle
arg_1_tensor = paddle.randint(-2,8192,[2, 2, 3, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.diagonal(arg_1,)
