import paddle
arg_1_0_tensor = paddle.rand([2, 23, 32], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([2, 23, 32], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = paddle.broadcast_tensors(arg_1,)
