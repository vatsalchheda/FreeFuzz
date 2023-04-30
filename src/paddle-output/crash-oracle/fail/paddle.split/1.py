import paddle
arg_1_tensor = paddle.randint(-4,8192,[3, 9, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 3
arg_2_2 = -1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = 1
res = paddle.split(arg_1,num_or_sections=arg_2,axis=arg_3,)
