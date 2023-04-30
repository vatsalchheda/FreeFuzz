import paddle
arg_1_tensor = paddle.randint(-32,512,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,128,[7], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-4096,4,[3], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
res = paddle.geometric.reindex_graph(arg_1,arg_2,arg_3,)
