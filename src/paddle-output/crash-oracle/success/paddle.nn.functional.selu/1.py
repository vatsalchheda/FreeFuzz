import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0507009873554805
arg_3 = 65.0
arg_4 = None
res = paddle.nn.functional.selu(arg_1,arg_2,arg_3,arg_4,)
