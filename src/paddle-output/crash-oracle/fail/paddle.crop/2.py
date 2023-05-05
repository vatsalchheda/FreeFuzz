import paddle
arg_1_tensor = paddle.randint(-8, 64, [3, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64, 1024, [2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.crop(arg_1,arg_2,)
