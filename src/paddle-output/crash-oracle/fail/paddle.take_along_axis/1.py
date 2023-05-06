import paddle
arg_1_tensor = paddle.rand([2, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0_0 = 0
arg_2_0_1 = 1
arg_2_0_2 = 2
arg_2_0 = [arg_2_0_0,arg_2_0_1,arg_2_0_2,]
arg_2_1_0 = 1
arg_2_1_1 = 2
arg_2_1_2 = 0
arg_2_1 = [arg_2_1_0,arg_2_1_1,arg_2_1_2,]
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = -11
res = paddle.take_along_axis(arg_1,arg_2,axis=arg_3,)
