import paddle
arg_1_0_tensor = paddle.randint(-8192,4,[2, 3], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-1,256,[2, 3], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = paddle.add_n(arg_1,)
