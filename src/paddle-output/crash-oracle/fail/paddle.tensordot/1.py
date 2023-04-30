import paddle
arg_1_tensor = paddle.randint(-8192,128,[3, 4, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,2048,[4, 3, 2], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3_0_0 = True
arg_3_0_1 = True
arg_3_0 = [arg_3_0_0,arg_3_0_1,]
arg_3_1_0 = "circular"
arg_3_1_1 = "max"
arg_3_1 = [arg_3_1_0,arg_3_1_1,]
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
