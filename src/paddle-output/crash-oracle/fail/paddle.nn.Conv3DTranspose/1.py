import paddle
arg_1 = -67
arg_2 = 6
arg_3_0 = 3
arg_3_1 = 3
arg_3_2 = 3
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
arg_class = paddle.nn.Conv3DTranspose(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([2, 4, 8, 8, 8], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = arg_class(*arg_4)
