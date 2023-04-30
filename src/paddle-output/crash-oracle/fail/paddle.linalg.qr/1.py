import paddle
arg_1_tensor = paddle.randint(-8192,512,[3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.qr(arg_1,)
