import paddle
arg_1_tensor = paddle.randint(-16,1024,[1], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
res = paddle.abs(arg_1,)
