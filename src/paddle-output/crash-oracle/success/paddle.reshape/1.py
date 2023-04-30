import paddle
arg_1_tensor = paddle.randint(-4096,128,[6], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = False
res = paddle.reshape(arg_1,arg_2,name=arg_3,)
