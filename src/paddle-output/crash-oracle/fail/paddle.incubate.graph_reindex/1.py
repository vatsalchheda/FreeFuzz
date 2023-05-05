import paddle
arg_1_tensor = paddle.rand([3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,128,[7], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16384,128,[3], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
res = paddle.incubate.graph_reindex(arg_1,arg_2,arg_3,)
