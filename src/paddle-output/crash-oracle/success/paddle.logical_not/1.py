import paddle
arg_1_tensor = paddle.randint(0,2,[1, 1, 1, 15])
arg_1 = arg_1_tensor.clone()
res = paddle.logical_not(arg_1,)
