import paddle
arg_1_tensor = paddle.randint(-16384,64,[60000], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3_tensor = paddle.randint(-8,8192,[60000], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
res = paddle.scatter(arg_1,arg_2,arg_3,)
