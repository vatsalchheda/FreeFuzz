import paddle
arg_1_tensor = paddle.rand([2, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 45
arg_2_1 = -57
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.moveaxis(arg_1,arg_2,arg_3,)
