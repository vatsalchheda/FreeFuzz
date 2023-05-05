import paddle
arg_1_tensor = paddle.randint(-32,1024,[13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,128,[11], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1024,32768,[4], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4 = 1
res = paddle.geometric.sample_neighbors(arg_1,arg_2,arg_3,sample_size=arg_4,)
