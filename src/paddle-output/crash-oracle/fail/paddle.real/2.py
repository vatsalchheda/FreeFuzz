import paddle
arg_1_tensor = paddle.randint(-32768,16,[2, 3], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.real(arg_1,)
