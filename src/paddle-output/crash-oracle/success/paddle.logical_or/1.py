import paddle
arg_1_tensor = paddle.randint(0,2,[2, 1])
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[29])
arg_2 = arg_2_tensor.clone()
res = paddle.logical_or(arg_1,arg_2,)
