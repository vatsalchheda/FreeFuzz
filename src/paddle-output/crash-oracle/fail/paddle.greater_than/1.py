import paddle
arg_1_tensor = paddle.randint(-1, 32, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([4], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.greater_than(arg_1,arg_2,)
