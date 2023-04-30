import paddle
arg_1_tensor = paddle.randint(-2,64,[2, 4], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.sgn(arg_1,)
