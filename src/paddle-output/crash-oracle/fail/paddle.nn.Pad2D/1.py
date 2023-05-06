import paddle
arg_1_0 = -64
arg_1_1 = -30
arg_1_2 = 38
arg_1_3 = 36
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = -9980.0
arg_class = paddle.nn.Pad2D(arg_1,value=arg_2,)
arg_3_0_tensor = paddle.rand([1, 1, 89, 88], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
