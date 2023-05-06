import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16, 2048, [3], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
res = paddle.incubate.segment_mean(arg_1,arg_2,)
