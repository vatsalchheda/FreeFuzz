import paddle
arg_1_tensor = paddle.randint(-4096,32768,[1, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.ceil(arg_1,)
