import paddle
arg_1_tensor = paddle.randint(-16,16384,[3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,16384,[4, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.mm(arg_1,arg_2,)
