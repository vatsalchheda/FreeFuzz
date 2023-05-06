import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048, 256, [5], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.greater_equal(arg_1,arg_2,)
