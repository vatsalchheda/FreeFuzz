import paddle
arg_1_tensor = paddle.randint(-2048,32,[2, 3], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.as_real(arg_1,)
