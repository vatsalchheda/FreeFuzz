import paddle
arg_1_tensor = paddle.randint(-32, 16, [3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.sparse.deg2rad(arg_1,)
