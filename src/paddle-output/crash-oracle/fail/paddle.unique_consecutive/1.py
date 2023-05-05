import paddle
arg_1_tensor = paddle.randint(-256,8192,[4, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.unique_consecutive(arg_1,axis=arg_2,)
