import paddle
arg_1_tensor = paddle.randint(-4096,32,[3, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = -27
res = paddle.argmin(arg_1,axis=arg_2,)
