import paddle
arg_1_tensor = paddle.randint(0,2,[7])
arg_1 = arg_1_tensor.clone()
res = paddle.isfinite(arg_1,)
