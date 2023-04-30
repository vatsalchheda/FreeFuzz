import paddle
arg_1_tensor = paddle.randint(-128,32768,[3, 1024], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.eig(arg_1,)
