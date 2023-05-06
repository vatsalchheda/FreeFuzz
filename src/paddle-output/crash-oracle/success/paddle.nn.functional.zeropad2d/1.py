import paddle
arg_1_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -43
arg_2_1 = -29
arg_2_2 = 41
arg_2_3 = -62
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
res = paddle.nn.functional.zeropad2d(arg_1,arg_2,)
