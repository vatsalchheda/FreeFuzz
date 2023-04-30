import paddle
arg_1_tensor = paddle.randint(-1024,4,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,8192,[2, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fmin(arg_1,arg_2,)
