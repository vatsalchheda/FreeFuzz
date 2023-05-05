import paddle
arg_1_tensor = paddle.rand([100, 3, 224, 224], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -7.0
arg_2_1 = "replicate"
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 1
arg_3_1 = 1
arg_3_2 = 2
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
arg_4 = -11.0
arg_5 = 1
res = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
