import paddle
arg_1_tensor = paddle.randint(-8,16384,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,32768,[2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 1
arg_4_tensor = paddle.randint(-16,1,[2, 3], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
res = paddle.index_add(arg_1,arg_2,arg_3,arg_4,)
