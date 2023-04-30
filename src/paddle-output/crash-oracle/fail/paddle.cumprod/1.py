import paddle
arg_1_tensor = paddle.randint(-4096,2048,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "circular"
arg_3 = "float64"
res = paddle.cumprod(arg_1,dim=arg_2,dtype=arg_3,)
