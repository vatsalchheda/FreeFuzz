import paddle
arg_1_tensor = paddle.randint(-256,1024,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -11
arg_2_1 = 8
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.expand(arg_1,arg_2,)
