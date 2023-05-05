import paddle
arg_1_tensor = paddle.rand([4, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.8
arg_3 = 1
arg_4 = True
res = paddle.quantile(arg_1,q=arg_2,axis=arg_3,keepdim=arg_4,)
