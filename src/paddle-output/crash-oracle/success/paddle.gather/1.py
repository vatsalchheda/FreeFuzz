import paddle
arg_1_tensor = paddle.randint(-1,128,[3, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,4,[2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = -5
res = paddle.gather(arg_1,arg_2,axis=arg_3,)
