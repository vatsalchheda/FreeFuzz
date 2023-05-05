import paddle
arg_1_tensor = paddle.rand([0, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 14.0
arg_3 = True
arg_4 = 2.05
res = paddle.renorm(arg_1,arg_2,arg_3,arg_4,)
