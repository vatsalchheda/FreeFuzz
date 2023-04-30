import paddle
arg_1_tensor = paddle.randint(-512,32,[3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,128,[4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16384,8192,[4, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "mean"
res = paddle.scatter(arg_1,arg_2,arg_3,overwrite=arg_4,)
