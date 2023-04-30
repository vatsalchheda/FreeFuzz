import paddle
arg_1_tensor = paddle.randint(-8,1,[2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -15
res = paddle.mode(arg_1,arg_2,)
