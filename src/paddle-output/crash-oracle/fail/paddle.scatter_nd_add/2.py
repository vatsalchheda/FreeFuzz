import paddle
arg_1_tensor = paddle.randint(-1024,2048,[3, 5, 9, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,1024,[3, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-512,16384,[3, 9, 10], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.scatter_nd_add(arg_1,arg_2,arg_3,)
