import paddle
arg_1_tensor = paddle.randint(-4096,16,[23, 51], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.cos(arg_1,)
