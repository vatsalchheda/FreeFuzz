import paddle
arg_1_tensor = paddle.randint(-8192,16384,[0, 43], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,16,[7], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_0_tensor = paddle.randint(-256,512,[3], dtype=paddle.int32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-4096,4096,[3], dtype=paddle.int32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.geometric.reindex_graph(arg_1,arg_2,arg_3,)
