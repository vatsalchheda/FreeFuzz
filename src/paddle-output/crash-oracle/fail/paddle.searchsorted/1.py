import paddle
arg_1_tensor = paddle.randint(-512, 512, [7], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384, 128, [2, 4], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
res = paddle.searchsorted(arg_1,arg_2,)
