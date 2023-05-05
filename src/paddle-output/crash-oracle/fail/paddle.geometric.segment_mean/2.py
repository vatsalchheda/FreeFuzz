import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2, 1, [3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.geometric.segment_mean(arg_1,arg_2,)
