import paddle
arg_1_tensor = paddle.randint(-16,128,[3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.min(arg_1,axis=arg_2,)
