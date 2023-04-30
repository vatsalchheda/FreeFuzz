import paddle
arg_1_tensor = paddle.randint(-16,16384,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,1024,[3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.geometric.segment_min(arg_1,arg_2,)
