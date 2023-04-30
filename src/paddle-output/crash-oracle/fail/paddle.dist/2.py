import paddle
arg_1_tensor = paddle.randint(-8192,8,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,512,[2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -inf
res = paddle.dist(arg_1,arg_2,arg_3,)
