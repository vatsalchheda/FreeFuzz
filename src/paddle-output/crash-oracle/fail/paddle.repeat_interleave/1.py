import paddle
arg_1_tensor = paddle.randint(-1, 8, [2, 3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512, 16384, [3], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
arg_3 = -1024
res = paddle.repeat_interleave(arg_1,arg_2,arg_3,)
