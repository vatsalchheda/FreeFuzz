import paddle
arg_1_tensor = paddle.randint(-8,2048,[3, 16], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,4,[3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.geometric.segment_max(arg_1,arg_2,)
