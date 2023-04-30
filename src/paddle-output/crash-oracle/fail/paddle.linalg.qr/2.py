import paddle
arg_1_tensor = paddle.randint(-8192,16384,[2, 3, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.qr(arg_1,)
