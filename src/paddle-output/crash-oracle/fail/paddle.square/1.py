import paddle
arg_1_tensor = paddle.randint(-16,8192,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.square(arg_1,)
