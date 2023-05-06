import paddle
arg_1_tensor = paddle.randint(-8, 256, [1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.deg2rad(arg_1,)
