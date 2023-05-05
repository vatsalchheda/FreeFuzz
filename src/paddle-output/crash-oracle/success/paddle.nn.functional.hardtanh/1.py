import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1.0
arg_3 = 3
arg_4 = "equal_nan"
res = paddle.nn.functional.hardtanh(arg_1,arg_2,arg_3,arg_4,)
