import paddle
arg_1_tensor = paddle.randint(-16384,32768,[2, 16], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.8
arg_3 = 0
res = paddle.quantile(arg_1,q=arg_2,axis=arg_3,)
