import paddle
arg_1_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "replicate"
arg_3 = 44.0
arg_4 = 0.75
arg_5 = 1.0
arg_6 = "sum"
arg_7 = None
res = paddle.nn.functional.local_response_norm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
