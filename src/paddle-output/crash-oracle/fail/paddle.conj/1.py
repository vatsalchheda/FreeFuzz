import paddle
arg_1_tensor = paddle.randint(-128,8192,[2, 3], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.conj(arg_1,)
