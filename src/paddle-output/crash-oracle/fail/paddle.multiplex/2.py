import paddle
arg_1_0_tensor = paddle.randint(-32,16,[2, 0], dtype=paddle.complex64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8,1,[2, 2], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2_tensor = paddle.randint(-1024,16384,[2, 1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.multiplex(arg_1,arg_2,)
