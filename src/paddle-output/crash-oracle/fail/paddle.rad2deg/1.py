import paddle
arg_1_tensor = paddle.randint(-4,8192,[6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.rad2deg(arg_1,)
