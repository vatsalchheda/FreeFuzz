import paddle
arg_1_tensor = paddle.randint(-256,32,[2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.diff(arg_1,axis=arg_2,)
