import paddle
arg_1_tensor = paddle.randint(-2048,128,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.cosh(arg_1,)
