import paddle
arg_1_tensor = paddle.randint(-4,128,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.exp(arg_1,)
