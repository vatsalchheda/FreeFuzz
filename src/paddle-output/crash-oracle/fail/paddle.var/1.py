import paddle
arg_1_tensor = paddle.randint(-1024,16,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.var(arg_1,axis=arg_2,)
