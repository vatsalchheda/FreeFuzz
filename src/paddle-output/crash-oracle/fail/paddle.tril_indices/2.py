import paddle
arg_1_tensor = paddle.randint(-2,256,[13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 8
arg_3 = 2
res = paddle.tril_indices(arg_1,arg_2,arg_3,)
