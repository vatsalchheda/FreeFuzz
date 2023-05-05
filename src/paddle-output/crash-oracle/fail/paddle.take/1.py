import paddle
arg_1_tensor = paddle.randint(-1,16384,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,16384,[3, 5], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = -50.0
res = paddle.take(arg_1,arg_2,mode=arg_3,)
