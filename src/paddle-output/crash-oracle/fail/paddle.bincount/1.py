import paddle
arg_1_tensor = paddle.randint(-4,512,[5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,128,[5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.bincount(arg_1,weights=arg_2,)
