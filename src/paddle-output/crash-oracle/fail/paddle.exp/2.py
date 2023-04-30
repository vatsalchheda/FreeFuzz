import paddle
arg_1_tensor = paddle.randint(-2048,256,[1, 2, 3, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.exp(arg_1,)
