import paddle
arg_1_tensor = paddle.randint(-512,32768,[3, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,2048,[2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32,8192,[2], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
res = paddle.crop(arg_1,arg_2,arg_3,)
