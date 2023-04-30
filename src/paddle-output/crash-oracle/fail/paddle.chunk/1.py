import paddle
arg_1_tensor = paddle.randint(-1,8,[3, 9, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 45
arg_3 = -2
res = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)
