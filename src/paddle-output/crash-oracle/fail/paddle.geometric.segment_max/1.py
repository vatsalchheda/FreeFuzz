import paddle
arg_1_tensor = paddle.randint(-4,32768,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,64,[3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.geometric.segment_max(arg_1,arg_2,)
