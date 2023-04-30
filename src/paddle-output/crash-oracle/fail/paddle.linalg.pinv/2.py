import paddle
arg_1_tensor = paddle.randint(-4096,8,[3, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.pinv(arg_1,)
