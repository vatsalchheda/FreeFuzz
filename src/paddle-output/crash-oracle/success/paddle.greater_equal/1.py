import paddle
arg_1_tensor = paddle.rand([2, 3, 1, 1, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.greater_equal(arg_1,arg_2,)
