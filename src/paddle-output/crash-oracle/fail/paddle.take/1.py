import paddle
arg_1_tensor = paddle.randint(-16384,32,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,2048,[2, 3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.take(arg_1,arg_2,)
