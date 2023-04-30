import paddle
arg_1_tensor = paddle.randint(-2,32768,[2, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,8192,[3], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.mv(arg_1,arg_2,)
