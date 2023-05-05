import paddle
arg_1_tensor = paddle.rand([0, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = 2
arg_4 = True
res = paddle.renorm(arg_1,arg_2,arg_3,arg_4,)
