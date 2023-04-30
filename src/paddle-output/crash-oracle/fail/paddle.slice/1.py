import paddle
arg_1_tensor = paddle.randint(-4096,128,[4, 4, 4], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2 = [arg_2_0,]
arg_3_0 = 0
arg_3 = [arg_3_0,]
arg_4_0 = 3
arg_4_1 = 2
arg_4_2 = 4
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
res = paddle.slice(arg_1,arg_2,starts=arg_3,ends=arg_4,)
