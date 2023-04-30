import paddle
arg_1_tensor = paddle.randint(-2,8192,[4, 4], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.angle(arg_1,)
