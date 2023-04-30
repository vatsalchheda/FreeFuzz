import paddle
arg_1_tensor = paddle.randint(0,2,[2, 2], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = False
res = paddle.all(arg_1,axis=arg_2,keepdim=arg_3,)
