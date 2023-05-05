import paddle
arg_1_tensor = paddle.randint(-16384,4096,[13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,8192,[11], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-256,16384,[4], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4 = 2
res = paddle.incubate.graph_sample_neighbors(arg_1,arg_2,arg_3,sample_size=arg_4,)
