import paddle
arg_1_0 = -31
arg_1_1 = -33
arg_1_2 = 28
arg_1_3 = -29
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_class = paddle.nn.ZeroPad2D(padding=arg_1,)
arg_2_0_tensor = paddle.rand([1, 1, 2, 3], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
