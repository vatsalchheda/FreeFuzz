import paddle
arg_1_tensor = paddle.randint(-32,1024,[3, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.logical_not(arg_1,)
