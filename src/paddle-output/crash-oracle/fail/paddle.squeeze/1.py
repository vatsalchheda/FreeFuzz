import paddle
arg_1_tensor = paddle.rand([64, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 39
arg_2 = [arg_2_0,]
res = paddle.squeeze(arg_1,axis=arg_2,)
