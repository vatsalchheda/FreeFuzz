import paddle
arg_1_tensor = paddle.randint(-32768, 2048, [13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32, 64, [11], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1, 8192, [4], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4 = -54
res = paddle.geometric.sample_neighbors(arg_1,arg_2,arg_3,sample_size=arg_4,)
