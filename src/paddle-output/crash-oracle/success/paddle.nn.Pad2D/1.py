import paddle
arg_1_0 = 0
arg_1_1 = 0
arg_1_2 = 0
arg_1_3 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = "replicate"
arg_class = paddle.nn.Pad2D(arg_1,mode=arg_2,)
arg_3_0_tensor = paddle.rand([1, 1, 143, 143], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
