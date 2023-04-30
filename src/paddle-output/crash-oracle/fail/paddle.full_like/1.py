import paddle
arg_1_tensor = paddle.randint(-2,128,[1, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -9
res = paddle.full_like(arg_1,fill_value=arg_2,)
