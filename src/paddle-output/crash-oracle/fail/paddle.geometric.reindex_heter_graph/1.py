import paddle
arg_1_tensor = paddle.randint(-2048, 8192, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.randint(-1, 128, [7], dtype=paddle.int64arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-4, 4, [5], dtype=paddle.int64arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
arg_3_tensor = paddle.randint(-4, 16384, [3], dtype=paddle.int32arg_3 = arg_3_tensor.clone()
res = paddle.geometric.reindex_heter_graph(arg_1,arg_2,arg_3,)
