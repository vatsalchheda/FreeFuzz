import paddle
arg_1_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
arg_3 = 0.0001
arg_4 = 0.75
arg_5 = 1.0
arg_6 = "NCHW"
arg_7_0 = -62
arg_7_1 = 0
arg_7 = [arg_7_0,arg_7_1,]
res = paddle.nn.functional.local_response_norm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
