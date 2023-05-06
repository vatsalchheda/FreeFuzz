import paddle
arg_1_tensor = paddle.rand([2, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 19.3
arg_2_1 = 2.5
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 0
res = paddle.nanquantile(arg_1,q=arg_2,axis=arg_3,)
