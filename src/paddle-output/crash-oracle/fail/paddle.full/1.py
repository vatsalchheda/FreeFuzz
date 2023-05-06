import paddle
arg_1_0_tensor = paddle.randint(-1, 16384, [1], dtype=paddle.int32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8192, 4, [1], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "max"
arg_3 = "int32"
res = paddle.full(arg_1,arg_2,dtype=arg_3,)
