import paddle
arg_1_tensor = paddle.randint(-512, 8192, [2, 4], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
res = paddle.incubate.asp.calculate_density(arg_1,)
