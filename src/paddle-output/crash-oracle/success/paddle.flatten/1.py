import paddle
arg_1_tensor = paddle.rand([64, 16, 5, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.flatten(arg_1,arg_2,)
