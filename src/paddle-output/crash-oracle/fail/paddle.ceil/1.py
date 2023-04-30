import paddle
arg_1_tensor = paddle.randint(-2,512,[1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.ceil(arg_1,)
