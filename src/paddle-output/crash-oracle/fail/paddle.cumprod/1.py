import paddle
arg_1_tensor = paddle.randint(-4096, 1, [3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = False
res = paddle.cumprod(arg_1,dim=arg_2,)
