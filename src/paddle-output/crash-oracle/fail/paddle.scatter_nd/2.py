import paddle
arg_1_tensor = paddle.randint(-4,1,[3, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,1,[3, 9, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 60
arg_3_1 = -28
arg_3_2 = 67
arg_3_3 = 70
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
res = paddle.scatter_nd(arg_1,arg_2,arg_3,)
