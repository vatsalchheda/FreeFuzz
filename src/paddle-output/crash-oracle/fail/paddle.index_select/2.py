import paddle
arg_1_tensor = paddle.randint(-128,2048,[4, 2, 44, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,2048,[4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.index_select(arg_1,arg_2,)
