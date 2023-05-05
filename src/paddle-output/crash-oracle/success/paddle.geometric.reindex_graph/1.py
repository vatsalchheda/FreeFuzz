import paddle
arg_1_tensor = paddle.randint(-256, 2048, [3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512, 1, [7], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1024, 1, [3], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
res = paddle.geometric.reindex_graph(arg_1,arg_2,arg_3,)
