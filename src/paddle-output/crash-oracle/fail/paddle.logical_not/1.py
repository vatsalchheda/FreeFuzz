import paddle
arg_1_tensor = paddle.randint(-4,1,[4, 1], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
res = paddle.logical_not(arg_1,)
