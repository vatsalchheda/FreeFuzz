import paddle
arg_1_tensor = paddle.randint(-4096,64,[7], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.sqrt(arg_1,)
