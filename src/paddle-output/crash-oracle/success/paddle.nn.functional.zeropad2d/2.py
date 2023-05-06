import paddle
arg_1_tensor = paddle.rand([1, 1, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 2
arg_2_2 = 1
arg_2_3 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
res = paddle.nn.functional.zeropad2d(arg_1,arg_2,)
