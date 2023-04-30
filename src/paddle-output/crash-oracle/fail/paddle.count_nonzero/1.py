import paddle
arg_1_tensor = paddle.randint(-16384,32768,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.count_nonzero(arg_1,axis=arg_2,)
