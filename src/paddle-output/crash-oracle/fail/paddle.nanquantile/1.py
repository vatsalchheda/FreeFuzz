import paddle
arg_1_tensor = paddle.rand([2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = 50
arg_4 = True
res = paddle.nanquantile(arg_1,q=arg_2,axis=arg_3,keepdim=arg_4,)
