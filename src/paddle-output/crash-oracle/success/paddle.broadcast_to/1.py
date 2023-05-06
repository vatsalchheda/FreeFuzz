import paddle
arg_1_tensor = paddle.rand([4, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 4
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.broadcast_to(arg_1,arg_2,)
