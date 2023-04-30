import paddle
arg_1_tensor = paddle.randint(-16384,2048,[3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.eigvals(arg_1,)
