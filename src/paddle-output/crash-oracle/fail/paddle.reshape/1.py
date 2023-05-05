import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -64
arg_2_1 = -16
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.reshape(arg_1,arg_2,)
