import paddle
arg_1_tensor = paddle.randint(-1,8,[7], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,64,[2, 4], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.searchsorted(arg_1,arg_2,)
