import paddle
arg_1_tensor = paddle.randint(-4,1024,[2, 3, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.unstack(arg_1,axis=arg_2,)
