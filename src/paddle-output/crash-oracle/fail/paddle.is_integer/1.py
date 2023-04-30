import paddle
arg_1_tensor = paddle.randint(-512,8,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.is_integer(arg_1,)
