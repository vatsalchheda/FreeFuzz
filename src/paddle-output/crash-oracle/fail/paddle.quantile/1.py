import paddle
arg_1_tensor = paddle.rand([4, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3_0 = -41
arg_3_1 = -34
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.quantile(arg_1,q=arg_2,axis=arg_3,)
