import paddle
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 75.0
res = paddle.unstack(arg_1,arg_2,)
