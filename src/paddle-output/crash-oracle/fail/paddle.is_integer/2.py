import paddle
arg_1_tensor = paddle.randint(-256,16,[2], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.is_integer(arg_1,)
