import paddle
arg_1_tensor = paddle.randint(0,2,[1, 1])
arg_1 = arg_1_tensor.clone()
res = paddle.any(arg_1,)
