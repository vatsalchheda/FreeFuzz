import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -15.0
arg_3 = 1.0
arg_4 = None
res = paddle.nn.functional.hardtanh(arg_1,arg_2,arg_3,arg_4,)
