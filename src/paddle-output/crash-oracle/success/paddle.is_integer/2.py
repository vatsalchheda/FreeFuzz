import paddle
arg_1_tensor = paddle.randint(-2, 32768, [3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.is_integer(arg_1,)
