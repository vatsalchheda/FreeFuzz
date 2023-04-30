import paddle
arg_1_tensor = paddle.randint(-4,4096,[3, 2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.flip(arg_1,arg_2,)
