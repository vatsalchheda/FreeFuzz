import paddle
arg_1_tensor = paddle.randint(-8,1,[2, 5], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 1027.0
arg_3_1 = "mean"
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,shape=arg_3,)
