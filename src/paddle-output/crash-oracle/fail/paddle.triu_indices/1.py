import paddle
arg_1_tensor = paddle.randint(-8,64,[13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 4
arg_3 = -1
res = paddle.triu_indices(arg_1,arg_2,arg_3,)
