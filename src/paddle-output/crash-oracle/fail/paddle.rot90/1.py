import paddle
arg_1_tensor = paddle.randint(-256,128,[2, 2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3_0 = 41
arg_3_1 = -2
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.rot90(arg_1,arg_2,arg_3,)
