import paddle
arg_1_tensor = paddle.randint(-32,16,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([4, 3, 2], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.bitwise_or(arg_1,arg_2,)
