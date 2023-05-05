import paddle
arg_1_tensor = paddle.randint(-16, 16, [3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.not_equal(arg_1,arg_2,)
