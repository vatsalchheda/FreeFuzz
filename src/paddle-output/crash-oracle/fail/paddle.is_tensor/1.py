import paddle
arg_1_tensor = paddle.randint(-16, 32, [5, 20], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.is_tensor(arg_1,)
