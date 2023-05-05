import paddle
arg_1_tensor = paddle.randint(-1, 256, [1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.shape(arg_1,)
