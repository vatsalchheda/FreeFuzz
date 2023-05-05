import paddle
arg_1_tensor = paddle.randint(-512, 16, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512, 256, [12], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1, 16384, [6], dtype=paddle.int32arg_3 = arg_3_tensor.clone()
res = paddle.incubate.graph_reindex(arg_1,arg_2,arg_3,)
