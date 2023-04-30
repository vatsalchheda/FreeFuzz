import paddle
arg_1_tensor = paddle.randint(-256,2048,[5, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.incubate.asp.calculate_density(arg_1,)
