import paddle
arg_1_tensor = paddle.randint(-128, 8, [7], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1.0
arg_2_1 = 2.0
arg_2_2 = 3.0
arg_2_3 = 4.0
arg_2_4 = 5.0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
res = paddle.searchsorted(arg_1,arg_2,)
