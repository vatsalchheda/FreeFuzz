import paddle
arg_1_tensor = paddle.randint(-16384,64,[3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.svd(arg_1,)
