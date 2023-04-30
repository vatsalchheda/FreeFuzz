import paddle
arg_1_tensor = paddle.randint(-32768,1024,[2, 2, 1, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
res = paddle.scale(arg_1,scale=arg_2,)
