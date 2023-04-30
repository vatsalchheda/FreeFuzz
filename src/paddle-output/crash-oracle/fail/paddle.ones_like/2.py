import paddle
arg_1_tensor = paddle.randint(-64,32768,[1, 102, 1666], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.ones_like(arg_1,)
