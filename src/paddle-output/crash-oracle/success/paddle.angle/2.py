import paddle
arg_1_tensor = paddle.rand([32, 11], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.angle(arg_1,)
