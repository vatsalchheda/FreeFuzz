import paddle
arg_1_tensor = paddle.randint(-2048,32,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 63.0
res = paddle.scale(arg_1,scale=arg_2,)
