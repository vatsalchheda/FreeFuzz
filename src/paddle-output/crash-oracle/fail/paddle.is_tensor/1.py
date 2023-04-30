import paddle
arg_1_tensor = paddle.randint(-16384,2048,[1, 1, 2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.is_tensor(arg_1,)
