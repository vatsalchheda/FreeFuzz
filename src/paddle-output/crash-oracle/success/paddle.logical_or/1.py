import paddle
arg_1_tensor = paddle.randint(-1,256,[2, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,1,[2, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.logical_or(arg_1,arg_2,)
