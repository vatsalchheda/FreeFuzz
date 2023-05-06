import paddle
arg_1_tensor = paddle.rand([4, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2.5
arg_3 = 62
res = paddle.quantile(arg_1,q=arg_2,axis=arg_3,)
