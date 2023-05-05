import paddle
arg_1_tensor = paddle.rand([6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 7
arg_2_1 = 60
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = False
res = paddle.sum(arg_1,axis=arg_2,keepdim=arg_3,)
