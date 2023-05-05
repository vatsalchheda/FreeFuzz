import paddle
arg_1_tensor = paddle.randint(-8, 8, [1, 5], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 5
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.broadcast_to(arg_1,arg_2,)
