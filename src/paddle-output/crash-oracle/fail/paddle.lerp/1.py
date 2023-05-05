import paddle
arg_1_tensor = paddle.rand([2, 3, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1, 2], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([1, 2], dtype=paddle.float64)
arg_3 = arg_3_tensor.clone()
res = paddle.lerp(arg_1,arg_2,arg_3,)
