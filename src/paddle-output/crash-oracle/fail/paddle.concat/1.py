import paddle
arg_1_0_tensor = paddle.randint(-1024, 4, [1, 11], dtype=paddle.int64arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-1024, 512, [1, 1], dtype=paddle.int64arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 1
res = paddle.concat(arg_1,axis=arg_2,)
