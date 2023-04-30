import paddle
arg_1_tensor = paddle.randint(-32,8,[3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -7
res = paddle.logcumsumexp(arg_1,axis=arg_2,)
