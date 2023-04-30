import paddle
arg_1_tensor = paddle.randint(-4,8192,[1, 64, 324], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.log10(arg_1,)
