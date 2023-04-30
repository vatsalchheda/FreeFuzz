import paddle
arg_1_tensor = paddle.randint(-2,32,[6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.deg2rad(arg_1,)
