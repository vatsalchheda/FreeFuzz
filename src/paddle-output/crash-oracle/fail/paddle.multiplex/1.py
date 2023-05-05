import paddle
arg_1_0_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2_tensor = paddle.randint(-1024, 2048, [2, 1], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
res = paddle.multiplex(arg_1,arg_2,)
