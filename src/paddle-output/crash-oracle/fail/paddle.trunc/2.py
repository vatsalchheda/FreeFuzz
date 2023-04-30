import paddle
arg_1_tensor = paddle.randint(-64,32,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.trunc(arg_1,)
