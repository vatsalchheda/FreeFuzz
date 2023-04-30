import paddle
arg_1_tensor = paddle.randint(-16384,16384,[2, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,256,[2, 2], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = False
res = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
