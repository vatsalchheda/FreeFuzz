import paddle
arg_1_tensor = paddle.randint(-8192,8192,[2, 1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 1
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.broadcast_to(arg_1,arg_2,)
