import paddle
arg_1_tensor = paddle.rand([4, 2, 2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384, 16384, [4], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.index_select(arg_1,arg_2,)
