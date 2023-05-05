import paddle
arg_1_tensor = paddle.rand([2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1024
arg_2 = [arg_2_0,]
arg_3 = False
arg_4 = None
res = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
