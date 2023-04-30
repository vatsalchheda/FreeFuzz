import paddle
arg_1_tensor = paddle.randint(-8192,2048,[7], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.isinf(arg_1,)
