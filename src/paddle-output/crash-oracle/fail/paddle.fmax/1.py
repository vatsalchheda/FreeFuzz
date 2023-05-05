import paddle
arg_1_tensor = paddle.randint(-4096, 16384, [2, 2], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32, 64, [2, 2], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.fmax(arg_1,arg_2,)
