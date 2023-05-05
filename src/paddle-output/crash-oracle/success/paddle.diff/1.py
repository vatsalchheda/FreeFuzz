import paddle
arg_1_tensor = paddle.randint(-32768,1024,[2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -32
res = paddle.diff(arg_1,axis=arg_2,)
