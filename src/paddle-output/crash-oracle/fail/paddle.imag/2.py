import paddle
arg_1_tensor = paddle.randint(-32,128,[2, 3], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.imag(arg_1,)
