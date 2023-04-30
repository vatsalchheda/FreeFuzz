import paddle
arg_1_tensor = paddle.randint(-8192,64,[2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.argsort(arg_1,axis=arg_2,)
