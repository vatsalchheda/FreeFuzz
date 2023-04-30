import paddle
arg_1_tensor = paddle.randint(-64,32,[2, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.min(arg_1,)
