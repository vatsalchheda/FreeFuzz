import paddle
arg_1_tensor = paddle.randint(-8, 1024, [3, 3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16, 1024, [2], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16, 2, [2], dtype=paddle.int32arg_3 = arg_3_tensor.clone()
res = paddle.crop(arg_1,arg_2,arg_3,)
