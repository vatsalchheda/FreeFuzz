import paddle
arg_1_tensor = paddle.randint(-1,16384,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,4096,[2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 9.5
res = paddle.dist(arg_1,arg_2,arg_3,)
