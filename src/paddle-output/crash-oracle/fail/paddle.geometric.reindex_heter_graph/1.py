import paddle
arg_1_tensor = paddle.randint(-16384,64,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.randint(-8192,8,[7], dtype=paddle.int64)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-16,64,[5], dtype=paddle.int64)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0_tensor = paddle.randint(-1024,4,[], dtype=paddle.int32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-1,256,[3], dtype=paddle.int32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.geometric.reindex_heter_graph(arg_1,arg_2,arg_3,)
