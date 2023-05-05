import paddle
arg_1_tensor = paddle.randint(-1024,16,[3, 2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -40
res = paddle.flip(arg_1,arg_2,)
